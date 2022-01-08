/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include "codegen_hlsc.h"
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <regex>
#include <string>
#include <vector>
#include "../build_common.h"

namespace TVM {
namespace codegen {

void CodeGenHLSC::AddFunction(LoweredFunc f,
                              str2tupleMap<std::string, Type> map_arg_type) {
  CodeGenC::AddFunction(f, map_arg_type);
}

std::string CodeGenHLSC::GetBufferRef(Type t, const Variable* buffer,
                                      Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  if (t.lanes() == 1) {
    bool is_scalar =
        (buf_length_map_.count(buffer) == 1 && buf_length_map_[buffer] == 1);
    if (is_scalar) {
      os << vid;
    } else {
      os << vid;
      CHECK(var_shape_map_.count(buffer))
          << "buffer " << buffer->name_hint << " not found in var_shape_map";
      std::vector<Expr> indices =
          ExtractIndices(index, var_shape_map_[buffer], range_);
      for (size_t i = 0; i < indices.size(); i++) {
        os << '[';
        PrintExpr(indices[i], os);
        os << ']';
      }
    }
  }
  return os.str();
}

void CodeGenHLSC::VisitExpr_(const Min* op, std::ostream& os) {  // NOLINT(*)
  os << "std::min(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenHLSC::VisitExpr_(const Max* op, std::ostream& os) {  // NOLINT(*)
  os << "std::max(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenHLSC::VisitStmt_(const LetStmt* op) { CodeGenC::VisitStmt_(op); }

void CodeGenHLSC::VisitStmt_(const For* op) {
  // ignore the data tranmission for stmts
  if (const For* for_op = op->body.as<For>()) {
    while (for_op->body.as<For>()) for_op = for_op->body.as<For>();
    if (for_op->body.as<StreamStmt>()) {
      return;
    } else if (auto st = for_op->body.as<Store>()) {
      if (st->value.as<StreamExpr>()) return;
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenHLSC::GenForStmt(const For* op, std::string pragma, bool before) {
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
        if (label->value == "")
          stream << vid;
        else
          stream << label->value << "_" << vid;
        stream << ": ";
        break;
      }
    }
  }
  if (!loop_stage_name) stream << vid << ": ";
  stream << "for (";
  PrintType(op->loop_var.type(), stream);
  stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid
         << ") {\n";
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

void CodeGenHLSC::VisitStmt_(const IfThenElse* op) {
  std::string cond = PrintExpr(op->condition);
  // Skip the buffer data checking
  if (std::regex_match(cond, std::regex("!\\((arg)(.+)(== NULL)\\)"))) return;
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

void CodeGenHLSC::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid;
  if (!var_idmap_.count(op->buffer_var.get()))
    vid = AllocVarID(op->buffer_var.get());
  else
    vid = GetVarID(op->buffer_var.get());
  this->PrintIndent();
  int32_t constant_size = op->constant_allocation_size();
  CHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";
  const Variable* buffer = op->buffer_var.as<Variable>();
  var_shape_map_[buffer] = op->extents;

  std::string scope;  // allocate on local scope by default
  auto it = alloc_storage_scope_.find(buffer);
  if (it != alloc_storage_scope_.end())
    scope = alloc_storage_scope_.at(buffer);
  else
    scope = "local";
  PrintStorageScope(scope, stream);

  // hard fix alloc for channel
  if (vid.find("stream_") == std::string::npos) {
    if (op->is_const) stream << "const ";
    PrintType(op->type, stream);
    stream << ' ' << vid;
    if (constant_size > 1) {  // Transfer length one array to scalar
      stream << "[";
      for (size_t i = 0; i < op->extents.size(); i++) {
        PrintExpr(op->extents[i], stream);
        if (i != op->extents.size() - 1) stream << " * ";
      }
      stream << "]";
    }
  }
  buf_length_map_[buffer] = constant_size;
  RegisterHandleType(op->buffer_var.get(), op->type);
  for (size_t i = 0; i < op->attrs.size(); i++) {
    this->PrintStmt(op->attrs[i]);
  }
  this->PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace TVM
