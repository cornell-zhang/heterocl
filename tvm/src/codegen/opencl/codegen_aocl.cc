#include <tvm/ir_pass.h>
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_aocl.h"
#include "../../runtime/thread_storage_scope.h"

namespace TVM {
namespace codegen {

inline Type String2Type(std::string& s) {
  if (s.front() == '\"' && s.back() == '\"') {
    s.erase(0, 1);
    s.pop_back();
  }
  std::istringstream is(s);
  halideir_type_code_t code = Type::Int;
  if (s.substr(0, 3) == "int") {
    code = Type::Int; s = s.substr(3);
  } else if (s.substr(0, 4) == "uint") {
    code = Type::UInt; s = s.substr(4);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);
  } else if (s == "handle") {
    return Handle();
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  int bits = 32, lanes = 1;
  if (sscanf(s.c_str(), "%dx%d", &bits, &lanes) == 0) {
    LOG(FATAL) << "unknown type " << s;
  }
  return Type(code, bits, lanes);
}

void CodeGenAOCL::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // Clear previous generated state
  this->InitFuncState(f);
  for (Var arg: f->args) {
      if (arg.type().is_handle()) {
          alloc_storage_scope_[arg.get()] = "global";
      }
  }

  // Skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");

  // Register alloc buffer type
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  this->decl_stream << "#include \"ihc_apint.h\"" << "\n";
  this->decl_stream << "#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable\n";
  this->stream << "__kernel " << "void " << f->name << "(";

  // Write arguments
  for (size_t i = 0; i < f->args.size(); ++i) {
    // alloc or get var name
    Var v = f->args[i];
    std::string vid;
    if (!var_idmap_.count(v.get())) 
      vid = AllocVarID(v.get());
    else vid = GetVarID(v.get());

    if (i != 0) this->stream << ", ";
    if (map_arg_type.find(vid) == map_arg_type.end()) {
      LOG(WARNING) << vid << " type not found\n";
      PrintType(v.type(), this->stream);
      this->stream << ' ' << vid;
    }
    else {
      auto arg = map_arg_type[vid];
      this->stream << "__global ";
      PrintType(std::get<1>(arg), this->stream);
      if (v.type().is_handle())
        this->stream << "*";
      this->stream << ' ' << "restrict ";
      this->stream << std::get<0>(arg);
    }
  }
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  // this->stream << ' '<< ' ' << "return;\n";
  this->stream << "}\n\n";
}

void CodeGenAOCL::PrintType(Type t, std::ostream &os)
{
  int lanes = t.lanes();
  if(t.is_handle()) {
    os << "void*";return;
  }
  if(t == Bool()) {
    os <<"bool"; return;
  }
  CHECK_EQ(lanes, 1)
      << "do not yet support vector types";
  
  bool fail = false;
  if(t.is_float()) {
    switch(t.bits())
    {
      case 16:
        os << "half";
        // enable_fp16_ = true;
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        // enable_fp64_ = true;
        break;
      default:
        fail = true;
        break;
    }
    if(!fail && lanes ==1) return;
    if(!fail && (lanes >= 2 && lanes <=16))
    {
      os<<lanes; return;
    }

  // integer data type
  } else if(t.is_uint() || t.is_int()) {
    fail = true;
    if(!fail && lanes == 1) return;
    if(!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes; return;
    }
    if(fail && lanes==1) {
      if(t.is_uint()) {
        switch(t.bits()) {
          case 8:
            os << "uint8";
            return;
          case 16:
            os << "uint16";
            return;
          case 32:
            os << "uint";
            return;
          default:
            os << "ap_uint<" << t.bits() << ">"; 
            return;
        }
      }
      if(t.is_int()) {
        switch(t.bits()) {
          case 8:
            os << "int8";
            return;
          case 16:
            os << "int16";
            return;
          case 32:
            os << "int";
            return;
          default:
            os << "ap_int<" << t.bits() << ">"; 
            return;
        }
      }
    }
  }

  LOG(FATAL) << "Cannot convert type"<<t<<"to AOCL type";
}

void CodeGenAOCL::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  if (op->new_expr.defined()) {
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* "<< vid << '=' << new_data << ";\n";
  } else {
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    const Variable* buffer = op->buffer_var.as<Variable>();
    var_shape_map_[buffer] = op->extents;

    std::string scope; // allocate on local scope by default 
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end())
      scope = alloc_storage_scope_.at(buffer);
    else scope = "local";

    // ptr mode: use same naming for .new tensor
    if (vid.find("_new") != std::string::npos) {
      vid.replace(vid.find("_new"), 4, "");
      var_idmap_[op->buffer_var.get()] = vid; 
    }

    // not allocate buffer for channel or moved data
    if (alloc_set_.find(vid) == alloc_set_.end()) {
      this->PrintIndent();

      // allocate stream channels 
      if (vid.find("_pipe") != std::string::npos) {
        decl_stream << "channel ";
        PrintType(op->type, decl_stream);
        decl_stream << " " << vid << ";\n";
      } else {
        PrintType(op->type, stream);

        stream << ' '<< vid;
        if (constant_size > 1) { // Transfer length one array to scalar
          if (vid.find("_reuse") != std::string::npos) {
            for (size_t i = 0; i < op->extents.size(); i++) {
              stream << '[';
              PrintExpr(op->extents[i], stream);
              stream << "]";
            }
          } else {
            stream << '[' << constant_size << "]";
          }
        }
        stream << ";\n";
      }
      // pragmas associated with allocate 
      // for (auto& k : op->attrs) {
      //   if (!k.as<StreamStmt>()) this->PrintStmt(k);
      // }
    }
    buf_length_map_[buffer] = constant_size;
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenAOCL::VisitStmt_(const For* op) {
  std::ostringstream os;

  // always treat channel as ptr
  Stmt stmt = op->body;
  while (const For* for_op = stmt.as<For>())
    stmt = for_op->body;

  if (auto s = stmt.as<StreamStmt>()) { 
    if (s->buffer_var.get()->name_hint.find("channel") 
        != std::string::npos) return;
  } else if (auto st = stmt.as<Store>()) {
    if (auto e = st->value.as<StreamExpr>()) {
      if (e->buffer_var.get()->name_hint.find("channel")
          != std::string::npos) return;

    } else { 
      auto value = st->value;
      if (auto c = value.as<Cast>()) value = c->value;
      if (auto v = value.as<IntImm>()) {
        if (v->value == 0) return;
      } else if (auto v = value.as<FloatImm>()) {
        if (v->value == 0) return;
      } else if (auto v = value.as<UIntImm>()) {
        if (v->value == 0) return;
      }
    }
  }

  if (op->for_type == ForType::Unrolled) {
    int unroll_factor = 0, i = 0;
    for (auto key : op->annotate_keys) {
      if (auto str = key.as<StringImm>()) {
        auto factor = op->annotate_values[i].as<IntImm>();
        if (str->value == "factor" && factor != nullptr && factor->value > 1) {
          unroll_factor = factor->value;
          break;
        }
      }
      i++;
    }
    os << "#pragma unroll";
    if (unroll_factor > 0) os << " " << unroll_factor << "\n";
    else                   os << "\n";
  }
  else if (op->for_type == ForType::Pipelined) {
    int II = 1, i = 0;
    for (auto key : op->annotate_keys) {
      if (auto str = key.as<StringImm>()) {
        auto initiation_interval = op->annotate_values[i].as<IntImm>();
        if (str->value == "initiation_interval" &&
            initiation_interval != nullptr &&
            initiation_interval->value > 1) {
          II = initiation_interval->value;
          break;
        }
      }
      i++;
    }
    os << "#pragma";
    os << " ii " << II << "\n";
  }
  CodeGenAOCL::GenForStmt(op, os.str(), true);
}

void CodeGenAOCL::VisitExpr_(const StreamExpr* op, std::ostream& os) {
  std::string vid;
  if (!var_idmap_.count(op->buffer_var.get())) 
    vid = AllocVarID(op->buffer_var.get());
  else vid = GetVarID(op->buffer_var.get());
  int i = 0;
  for (auto key : op->annotate_keys) {
    auto str = key.as<StringImm>();
    auto val = op->annotate_values[i].as<StringImm>();
    if (str->value == "name" && val != nullptr) {
        vid = val->value;
        decl_stream << "channel ";
        PrintType(op->type, decl_stream);
        decl_stream << " " << vid << ";\n";
    }
    i++;
  }
  switch (op->stream_type) {
    case StreamType::FIFO:
      os << "read_channel_intel(";
      os << vid << ")";
      break;
  }
}

void CodeGenAOCL::VisitStmt_(const KernelDef* op) {
  LoweredFunc f;
  SaveFuncState(f);
  InitFuncState(f);
  std::ostringstream save;
  save << this->stream.str();
  this->stream.str("");
  this->stream.clear();

  // skip the first underscore
  GetUniqueName("_");
  // add to alloc buffer : type.
  for (const auto & k : op->args) {
    RegisterHandleType(k.get(), k.get()->type);
  }

  stream << "__kernel ";
  const UIntImm* is_void = op->ret_void.as<UIntImm>();
  if (is_void) stream << "void";
  else PrintType(op->ret_type, stream);
  stream << " " << op->name << "(";

  // top-level function 
  if (op->name == "test") {

    for (size_t i = 0; i < op->args.size(); ++i) {
      VarExpr v = op->args[i];
      var_shape_map_[v.get()] = op->arg_shapes[i];
      std::string vid = AllocVarID(v.get());

      // for top kernel functions 
      if (vid.find("_channel")) {
        vid.replace(vid.find("_channel"), 8, "");
        alloc_set_.insert(vid);
        alloc_set_.insert(vid + "_new");
      }

      if (i != 0) stream << ", ";
      this->stream << "__global ";
      std::string str = PrintExpr(op->arg_types[i]);
      Type type = String2Type(str);
      PrintType(type, stream);
      this->stream << "* restrict " << vid;
    }  
    stream << ") {\n";
    int func_scope = BeginScope();
    range_ = CollectIterRange(op->body);
    PrintStmt(op->body);
    EndScope(func_scope);
    stream << "}\n\n";

  } else {

    // streamed arg position to channel index
    std::unordered_map<int, int> stream_args;
    for (size_t j = 0; j < op->channels.size(); j++) {
      auto info = op->channels[j];
      int pos = info[0].as<IntImm>()->value;
      int idx = info[1].as<IntImm>()->value;
      stream_args[pos] = idx;
    } 

    for (size_t i = 0; i < op->args.size(); ++i) {
      VarExpr v = op->args[i];
      var_shape_map_[v.get()] = op->arg_shapes[i];
      std::string vid = AllocVarID(v.get());

      // for top kernel functions 
      if (vid.find("_channel")) {
        alloc_set_.insert(vid);
        vid.replace(vid.find("_channel"), 8, "");
        alloc_set_.insert(vid);
        alloc_set_.insert(vid + "_new");
      }

      if (stream_args.count(i)) { 
        stream_arg_pos[op->name].insert(i); 
        if (!stream_pragma) {
          decl_stream << "#pragma OPENCL EXTENSION cl_intel_channels : enable\n";
          stream_pragma = true;
        }
      } else {
        if (i != 0) {
          if (stream_args.count(i-1)) void(0);
          else stream << ", ";
        } // un-streamed argument 
        this->stream << "__global ";
        std::string str = PrintExpr(op->arg_types[i]);
        Type type = String2Type(str);
        PrintType(type, stream);
        this->stream << "* restrict " << vid;
      }
    }  
    stream << ") {\n";
    int func_scope = BeginScope();
    range_ = CollectIterRange(op->body);
    PrintStmt(op->body);
    EndScope(func_scope);
    stream << "}\n\n";
  }

  // restore default stream
  module_stream << this->stream.str();
  this->stream.str(""); 
  this->stream.clear();
  this->stream << save.str();
  RestoreFuncState(f);
}

void CodeGenAOCL::VisitStmt_(const KernelStmt *op) {
  PrintIndent();
  stream << op->name << "(";
  for (size_t i = 0; i < op->args.size(); i++) {
    std::string str = op->name + "." + PrintExpr(op->args[i]);
    if (!stream_arg_pos[op->name].count(i)) {
      if (i != 0) {
        if (stream_arg_pos[op->name].count(i-1)) void(0);
        else stream << ", ";
      }
      PrintExpr(op->args[i], stream);
    }
  }
  stream << ");\n";
}

void CodeGenAOCL::VisitExpr_(const KernelExpr *op, std::ostream& os) { // NOLINT(*)
  os << op->name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    if (!stream_arg_pos[op->name].count(i)) {
      if (i != 0) {
        if (stream_arg_pos[op->name].count(i-1)) void(0);
        else stream << ", ";
      }
      PrintExpr(op->args[i], stream);
    }
  }
  os << ")";
}

void CodeGenAOCL::VisitStmt_(const StreamStmt* op) {
  std::string vid;
  if (!var_idmap_.count(op->buffer_var.get())) 
    vid = AllocVarID(op->buffer_var.get());
  else vid = GetVarID(op->buffer_var.get());
  PrintIndent();
  int i = 0;
  for (auto key : op->annotate_keys) {
    auto str = key.as<StringImm>();
    auto val = op->annotate_values[i].as<StringImm>();
    if (str->value == "name" && val != nullptr) vid = val->value;
    i++;
  }
  switch (op->stream_type) {
    case StreamType::FIFO:
      stream << "write_channel_intel(";
      stream << vid << ", ";
      break;
  }
  PrintExpr(op->value, stream);
  stream << ");\n";
}

void CodeGenAOCL::VisitStmt_(const ExternModule* op) {
  std::string ip_name, func, header, deps, cmds;
  std::vector<std::string> args_in, args_out; 

  PrintIndent();
  for (size_t i = 0; i < op->annotate_keys.size(); i++) {
    auto key = op->annotate_keys[i].as<StringImm>()->value;
    if (key == "name") {
      ip_name = op->annotate_values[i].as<StringImm>()->value;
    } else if (key == "header") {
      header = op->annotate_values[i].as<StringImm>()->value;
    } else if (key == "func") {
      func = op->annotate_values[i].as<StringImm>()->value;
    } else if (key.find("input") != std::string::npos) { 
      auto arg = op->annotate_values[i].as<StringImm>()->value;
      args_in.push_back(arg);
    } else if (key.find("output") != std::string::npos) { 
      auto arg = op->annotate_values[i].as<StringImm>()->value;
      args_out.push_back(arg);
    } else if (key == "deps") { 
      cfg_stream << "deps: {" 
          << op->annotate_values[i].as<StringImm>()->value << "}\n";
    } else if (key == "cmds") { 
      cfg_stream << "cmds: {"
          << op->annotate_values[i].as<StringImm>()->value << "}\n";
    } else if (key == "flags") { 
      cfg_stream << "makefiles: {"
          << op->annotate_values[i].as<StringImm>()->value << "}\n";
    }
  }

  stream << func << "\n";
  // generate TCL and Makefile
  decl_stream << header << "\n";
}

} // namespace codegen
} // namespace TVM
