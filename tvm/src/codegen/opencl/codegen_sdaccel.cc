# include <tvm/runtime/config.h>
# include <tvm/packed_func_ext.h>
# include <tvm/ir_pass.h>
# include <vector>
# include <string>
# include "./codegen_sdaccel.h"
# include "../../runtime/thread_storage_scope.h"

namespace TVM {
namespace codegen {

void CodeGenSDACCEL::AddFunction(LoweredFunc f,
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
  this->stream << "__kernel " 
               << "void " << f->name << "(";

  // Write arguments
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
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
      this->stream << ' ' << std::get<0>(arg);
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

void CodeGenSDACCEL::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    //LOG(FATAL) << "The buffer shouldn't call PrintType for printing type";
    os << "void*";
    return ;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16: os << "half"; break;
      case 32: os << "float"; break;
      case 64: os << "double"; break;
      // case 128: os << "double double"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes; return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "unsigned ";
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int"; return;
    }

    int target_bit = 1;
    while (target_bit < t.bits())
      target_bit <<= 1;

    switch (target_bit) {
      case 1: os << "int"; break;
      case 2: os << "char"; break;
      case 4: os << "char"; break;
      case 8: os << "char"; break;
      case 16: os << "short"; break;
      case 32: os << "int"; break;
      case 64: os << "long"; break;
      case 128: os << "long"; break; // FIXME: Should use long long
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    // FIXME: Not yet support multiple lanes
    //if (!fail && (lanes >= 2 && lanes <= 16)) {
    //  os << lanes; return;
    //}
  }
  os << t;
  LOG(WARNING) << "Cannot convert type " << t ;
  return ;
}

void CodeGenSDACCEL::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  if (scope == "global" || scope == "shared") {
    os << "__local ";
  }
}

void CodeGenSDACCEL::VisitStmt_(const For* op) {
  std::ostringstream os;

  // ignore the data tranmission for stmts
  if (const For* for_op = op->body.as<For>()) {
    while (for_op->body.as<For>())
      for_op = for_op->body.as<For>();
    if (auto s = for_op->body.as<StreamStmt>()) { 
      if (s->buffer_var.get()->name_hint.find("channel") 
          != std::string::npos) return;
    } else if (auto st = for_op->body.as<Store>()) {
      if (auto e = st->value.as<StreamExpr>()) {
        if (e->buffer_var.get()->name_hint.find("channel")
            != std::string::npos) return;
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
    if (unroll_factor > 0) {
        os << "__attribute__((opencl_unroll_hint(";
        os << unroll_factor << ")))\n";
    } else {
      os << "\n";
    }
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
    os << "__attribute__((xcl_pipeline_loop(";
    os << II << ")))\n";
  }
  CodeGenSDACCEL::GenForStmt(op, os.str(), true);
}

void CodeGenSDACCEL::VisitStmt_(const Partition* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  stream << vid << " ";
  if (op->partition_type != PartitionType::Complete) {
    stream << "__attribute__((xcl_array_partition(";
    switch (op->partition_type) {
      case PartitionType::Complete:
        break;
      case PartitionType::Block:
        stream << "block,";
        break;
      case PartitionType::Cyclic:
        stream << "cyclic,";
        break;
      }
    stream << op->factor << ",";
    stream << op->dim << ")))\n";
  } else {
    if (op->dim == 0) {
      stream << "__attribute__((xcl_array_partition))\n";
    } else {
      stream << "__attribute__((xcl_array_partition(";
      stream << "complete,";
      stream << op->factor << ",";
      stream << op->dim << ")))\n";
      }
    }
}

void CodeGenSDACCEL::VisitStmt_(const Store* op) {
  if (auto e = op->value.as<StreamExpr>()) {
    // temp input to store data
    this->PrintIndent();
    stream << "int temp_in;\n";
    this->PrintIndent();
    stream << "read_pipe_block(" << GetVarID(e->buffer_var.get())
           << ", &temp_in);\n";

    std::string index = PrintExpr(op->index);
    this->PrintIndent();
    std::string vid = GetVarID(op->buffer_var.get());
    stream << vid << "[" << index << "] = temp_in;\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
};

void CodeGenSDACCEL::VisitStmt_(const Allocate* op) {
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

    std::string scope; // allocate on local scope by default 
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end())
      scope = alloc_storage_scope_.at(buffer);
    else scope = "local";

    // ignore channel and pipe buffers
    if (vid.find("c_buf_") == std::string::npos &&
        vid.find("channel") == std::string::npos) {
      this->PrintIndent();
      // PrintStorageScope(scope, stream);
      PrintType(op->type, stream);
      stream << ' '<< vid;
      if (constant_size > 1) // Transfer length one array to scalar
        stream << '[' << constant_size << "]";
      stream << ";\n";
    } else if (vid.find("c_buf_") != std::string::npos) { // register pipes
      if (!pipes.count(vid)) {
        pipes[vid] = 1; 
        decl_stream << "pipe int " << vid
                    << " __attribute__((xcl_reqd_pipe_depth(32)));\n"; 
      }
    }
    buf_length_map_[buffer] = constant_size;
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenSDACCEL::VisitStmt_(const StreamStmt* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  PrintIndent();
  switch (op->stream_type) {
    case StreamType::Channel:
      LOG(WARNING) << "not support channel in sdaccel; "
                   << "use pipe instead";
      break;
    case StreamType::FIFO:
      LOG(WARNING) << "not support fifo in sdaccel; "
                   << "use pipe instead";
      break;
    // declare outside def 
    case StreamType::Pipe:
      break;
  }
  stream << "int temp_out = "; 
  PrintExpr(op->value, stream);
  stream << ";\n";
  PrintIndent();
  stream << "write_pipe_block(" << vid
         << ", " << "&temp_out);\n";
}

void CodeGenSDACCEL::VisitExpr_(const StreamExpr* op, std::ostream& os) {
  std::string vid = GetVarID(op->buffer_var.get());
  os << vid << ".read()";
}

void CodeGenSDACCEL::VisitStmt_(const KernelDef* op) {
  // save func states
  LoweredFunc f;
  CodeGenC::SaveFuncState(f);
  CodeGenC::InitFuncState(f);
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

  // collect argument information 
  std::unordered_map<int, int> arg_info;
  for (size_t i = 0; i < op->channels.size(); i=i+2) {
    auto pos = op->channels[i].as<IntImm>()->value;
    auto idx = op->channels[i+1].as<IntImm>()->value;
    arg_info[pos] = idx;
  }
    
  // add function attribute and arguments 
  bool top_func = false;
  if (op->name.substr(0,13) == "top_function_") {
    top_func = true;
    stream << "__kernel\n";
    stream << "__attribute__((reqd_work_group_size(1, 1, 1)))\n";
    stream << "__attribute__((xcl_dataflow))\n";
  } else { // static sub function on kernel 
    stream << "static ";
  }

  stream << "void";
  // PrintType(op->ret_type, stream);
  stream << " " << op->name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    VarExpr v = op->args[i];
    var_shape_map_[v.get()] = op->api_args[i];
    std::string vid = AllocVarID(v.get());

    if (i != 0) stream << ", ";
    std::string str = PrintExpr(op->api_types[i]);
    Type type = String2Type(str);

    if (v.type().is_handle() && op->api_args[i].size() > 1) {
      if (top_func) this->stream << "__global ";
      PrintType(type, stream);
      this->stream << "* " << vid;
    } else {
      // PrintType(type, stream);
      this->stream << "int " << vid;
    }
  }  
  stream << ") {\n";
  int func_scope = BeginScope();
  range_ = CollectIterRange(op->body);
  PrintStmt(op->body);
  EndScope(func_scope);
  stream << "}\n\n";

  // restore default stream
  if (top_func) {
    module_stream << this->stream.str();
  } else { // put static function in the beginning
    std::ostringstream temp;
    temp << this->stream.str();
    temp << module_stream.str();
    module_stream.str("");
    module_stream.clear();
    module_stream << temp.str();
  }
  this->stream.str(""); 
  this->stream.clear();
  this->stream << save.str();
  RestoreFuncState(f);
}

} // namespace codegen
} // namespace TVM
