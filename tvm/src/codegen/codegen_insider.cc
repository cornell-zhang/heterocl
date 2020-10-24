/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_insider.cc
 */
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <tvm/ir_pass.h>
#include <vector>
#include <string>
#include <tuple>
#include <regex>
#include "./codegen_insider.h"
#include "../runtime/thread_storage_scope.h"

namespace TVM {
namespace codegen {

CodeGenInsider::CodeGenInsider() {
  restrict_keyword_ = "restrict"; // FIXME: Check if this is useful
  return ;
}

void CodeGenInsider::InitFuncState(LoweredFunc f) {
  CodeGenC::InitFuncState(f);
  for (Var arg : f->args) {
    if (arg.type().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
  return ;
}

void CodeGenInsider::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // Clear previous generated state
  this->InitFuncState(f);

  // Skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");

  // Register alloc buffer type
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  // Write header files
  this->stream << "#include <string.h>\n";
  this->stream << "#include <math.h>\n";
  this->stream << "#include <assert.h>\n";
  this->stream << "#include <insider_kernel.h>\n";
  this->stream << "##include \"constant.h\"\n";

  // Write entry function name
  this->stream << "void " << f->name << "(";

  // Write arguments
  // All args are Insider specific dtype
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
      this->stream << "ST_Queue<APP_Data> &";
      this->stream << std::get<0>(arg);
      top_args.insert(std::get<0>(arg));
    }
  }
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

std::string CodeGenInsider::Finish() {
  return CodeGenC::Finish();
}

void CodeGenInsider::BindThreadIndex(const IterVar& iv) {
  LOG(FATAL) << "Merlin doesn't support thread binding";
  return ;
}

void CodeGenInsider::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
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
      case 128: os << "double double"; break;
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

void CodeGenInsider::PrintVecAddr(const Variable* buffer, Type t,
                                 Expr base, std::ostream& os) {  // NOLINT(*)
  // FIXME: What's this node for?
  if (!HandleTypeMatch(buffer, t.element_of())) {
    os << '(';
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    os << ' ';
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer) << " + ";
  PrintExpr(base, os);
  return ;
}

void CodeGenInsider::PrintVecStore(const Variable* buffer,
                                  Type t, Expr base,
                                  const std::string& value) {
  // FIXME: What's this node for?
  this->PrintIndent();
  stream << "vstore" << t.lanes() << "(" << value << ", 0, ";
  PrintVecAddr(buffer, t, base, stream);
  stream << ");\n";
  return ;
}

void CodeGenInsider::PrintStorageSync(const Call* op) {
  const std::string& sync = op->args[0].as<StringImm>()->value;
  if (sync == "warp") {
    LOG(FATAL) << "warp sync not supported in Merlin";
  } else if (sync == "shared") {
    LOG(FATAL) << "shared sync not supported in Merlin";
  } else if (sync == "global") {
    LOG(FATAL) << "global sync not supported in Merlin";
  }
  return ;
}

void CodeGenInsider::VisitExpr_(const Load* op, std::ostream& os) {
  std::string vid = GetVarID(op->buffer_var.get());
  // TODO: find a betetr way to track streaming channels 
  if (top_args.find(vid) != top_args.end()) {
    PrintIndent(); 
    stream << vid << "_temp = " << vid << ".read_nb();\n";
    os << vid << "_temp.get_data()";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenInsider::VisitStmt_(const Store* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  if (top_args.find(vid) != top_args.end()) {
    auto value = PrintExpr(op->value);
    auto bits = handle_data_type_[op->buffer_var.get()].bits();
    PrintIndent(); 
    stream << "pkt_b" << bits << " " << vid <<  "_temp;\n";
    PrintIndent(); 
    stream << vid <<  "_temp.set_data(" << value << ");\n";
    PrintIndent(); 
    stream << vid <<  "_temp.set_keep(-1);\n";
    PrintIndent(); 
    stream << vid << ".write(" << vid << "_temp);\n";
    return;
  }

  // handle SetSlice
  if (const SetSlice* ss = op->value.as<SetSlice>()) {
    Type t = op->value.type();
    Expr new_index_left = ir::Simplify(ss->index_left - 1);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    std::string rhs = PrintExpr(ss->value);
    PrintIndent();
    this->stream << ref
                 << "(" << PrintExpr(new_index_left) << ", " << PrintExpr(ss->index_right)
                 << ") = " << rhs << ";\n";
  } else if (const SetBit* sb = op->value.as<SetBit>()) {
    Type t = op->value.type();
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent();
    this->stream << ref
                 << "[" << PrintExpr(sb->index)
                 << "] = " << PrintExpr(sb->value) << ";\n";
  } else if (auto expr_op = op->value.as<Select>()) {
    Type t = op->value.type();
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent();
    this->stream << "if (" << PrintExpr(expr_op->condition) << ") { \n";
    PrintIndent();
    this->stream << "  " << ref 
        << " = " << PrintExpr(expr_op->true_value) << ";\n";
    PrintIndent();
    this->stream << "} else { \n";
    PrintIndent();
    this->stream << "  " << ref 
        << " = " << PrintExpr(expr_op->false_value) << ";\n";
    PrintIndent();
    this->stream << "}\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenInsider::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
    return ;
}

void CodeGenInsider::VisitExpr_(const Broadcast* op, std::ostream& os) { // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->type, os);
  os << ")(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
  return ;
}

void CodeGenInsider::VisitStmt_(const LetStmt* op) {
  std::string value = PrintExpr(op->value);
  // Skip the argument retrieving assign statement
  std::string vid = AllocVarID(op->var.get());
  if (op->var.type() != Handle() &&
      value.find("TVMArray") == std::string::npos &&
      value.find("arg") != 0) {
    PrintIndent();
    PrintType(op->var.type(), this->stream);
    this->stream << ' '
                 << vid
                 << " = " << value << ";\n";
  }
  PrintStmt(op->body);
}

void CodeGenInsider::GenForStmt(const For* op, std::string pragma, bool before) {
  std::string extent = PrintExpr(op->extent);
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  if (before && pragma.length() > 0) {
    PrintIndent();
    stream << pragma;
  }
  PrintIndent();

  // print loop labels
  bool loop_stage_name = false;
  for (unsigned int i = 0; i < op->annotate_keys.size(); i++) {
    if (auto str = op->annotate_keys[i].as<StringImm>()) {
      if (str->value == "stage_name") {
        loop_stage_name = true;
        auto label = op->annotate_values[i].as<StringImm>();
        std::string output_label;
        if (label->value == "") {
          output_label = vid;
        } else {
          output_label = label->value + "_" + vid;
        }
        for (size_t i = 0; i < output_label.size(); ++i) {
          if (output_label[i] == '.') output_label[i] = '_';
        }
        stream << output_label << ": ";
        break;
      }
    }
  }
  if (!loop_stage_name)
    stream << vid << ": ";

  stream << "for (";
  PrintType(op->loop_var.type(), stream);
  stream << ' ' << vid << " = 0; "
            << vid << " < " << extent
            << "; ++" << vid << ") {\n";
  if (!before && pragma.length() > 0) {
    PrintIndent();
    stream << pragma;
  }
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenInsider::VisitStmt_(const For* op) {
  std::ostringstream os;

  Stmt stmt = op->body;
  while (const For* for_op = stmt.as<For>())
    stmt = for_op->body;

  // Skip for-loops for all 0 assignment 
  if (auto st = stmt.as<Store>()) {
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
    os << "#pragma HLS unroll";
    if (unroll_factor > 0) os << " factor=" << unroll_factor << "\n";
    else                   os << "\n";
  }
  else if (op->for_type == ForType::Pipelined) {
    int II = 0, i = 0;
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
    os << "#pragma HLS pipeline";
    if (II > 0) os << " II=" << II << "\n";
    else        os << "\n";
  }
  GenForStmt(op, os.str(), false);
}

void CodeGenInsider::VisitStmt_(const IfThenElse* op) {
  std::string cond = PrintExpr(op->condition);

  // Skip the buffer data checking
  if (std::regex_match(cond, std::regex("!\\((arg)(.+)(== NULL)\\)")))
      return ;

  PrintIndent();
  if (cond[0] == '(' && cond[cond.length() - 1] == ')') {
    stream << "if " << cond << " {\n";
  } else {
    stream << "if (" << cond << ") {\n";
  }
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);

  if (op->else_case.defined()) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case);
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}
}  // namespace codegen
}  // namespace TVM
