/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include "codegen_ihls.h"
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <string>
#include <vector>
#include "../build_common.h"

namespace TVM {
namespace codegen {

void CodeGenIntelHLS::AddFunction(
    LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type) {
  // Write header files
  this->stream << "#include <HLS/hls.h>\n";
  this->stream << "#include <HLS/ac_int.h>\n";
  this->stream << "#include <HLS/ac_fixed.h>\n";
  this->stream << "#include <HLS/ac_fixed_math.h>\n";
  this->stream << "#include <math.h>\n\n";
  this->stream << "component ";
  CodeGenHLSC::AddFunction(f, map_arg_type);
}

void CodeGenIntelHLS::PrintType(Type t, std::ostream& os) {
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.is_uint())
      os << "ac_int<" << t.bits() << ", false>";
    else if (t.is_int())
      os << "ac_int<" << t.bits() << ", true>";
    else if (t.is_ufixed())
      os << "ac_fixed<" << t.bits() << ", " << t.bits() - t.fracs()
         << ", false>";
    else
      os << "ac_fixed<" << t.bits() << ", " << t.bits() - t.fracs()
         << ", true>";
  } else {
    CodeGenC::PrintType(t, os);
  }
}

void CodeGenIntelHLS::VisitExpr_(const GetBit* op, std::ostream& os) {
  PrintExpr(op->a, os);
  os << "[";
  PrintExpr(op->index, os);
  os << "]";
}

void CodeGenIntelHLS::VisitExpr_(const GetSlice* op, std::ostream& os) {
  PrintExpr(op->a, os);
  Expr diff = ir::Simplify(op->index_left - op->index_right);
  const int64_t* val = as_const_int(diff);
  if (val == nullptr) LOG(FATAL) << "The bit selection range is not a constant";
  os << ".slc<" << *val << ">(";
  PrintExpr(op->index_right, os);
  os << ")";
}

void CodeGenIntelHLS::VisitStmt_(const Store* op) {
  // handle SetSlice
  if (const SetSlice* ss = op->value.as<SetSlice>()) {
    Type t = op->value.type();
    Expr new_index_left = ir::Simplify(ss->index_left - 1);
    Expr diff = ir::Simplify(ss->index_left - ss->index_right);
    const int64_t* val = as_const_int(diff);
    if (val == nullptr)
      LOG(FATAL) << "The bit selection range is not a constant";
    Type val_type = UInt(*val);
    Expr new_value = ir::Cast::make(val_type, ss->value);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent();
    this->stream << ref << ".set_slc(" << PrintExpr(ss->index_right) << ", "
                 << PrintExpr(new_value) << ");\n";
  } else if (const SetBit* sb = op->value.as<SetBit>()) {
    Type t = op->value.type();
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent();
    this->stream << ref << "[" << PrintExpr(sb->index)
                 << "] = " << PrintExpr(sb->value) << ";\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenIntelHLS::VisitStmt_(const For* op) {
  std::ostringstream os;
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
    if (unroll_factor > 0)
      os << " " << unroll_factor << "\n";
    else
      os << "\n";
  } else if (op->for_type == ForType::Pipelined) {
    int II = 1, i = 0;
    for (auto key : op->annotate_keys) {
      if (auto str = key.as<StringImm>()) {
        auto initiation_interval = op->annotate_values[i].as<IntImm>();
        if (str->value == "initiation_interval" &&
            initiation_interval != nullptr && initiation_interval->value > 1) {
          II = initiation_interval->value;
          break;
        }
      }
      i++;
    }
    os << "#pragma";
    os << " ii " << II << "\n";
  }
  GenForStmt(op, os.str(), true);
}

void CodeGenIntelHLS::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  this->PrintIndent();
  int32_t constant_size = op->constant_allocation_size();
  CHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";
  const Variable* buffer = op->buffer_var.as<Variable>();
  var_shape_map_[buffer] = op->extents;
  std::string scope = alloc_storage_scope_.at(buffer);
  PrintStorageScope(scope, stream);
  PrintType(op->type, stream);
  stream << ' ' << vid;
  if (constant_size > 1) {  // Transfer length one array to scalar
    for (size_t i = 0; i < op->extents.size(); i++) {
      stream << '[';
      PrintExpr(op->extents[i], stream);
      stream << "]";
    }
  }
  stream << ";\n";
  buf_length_map_[buffer] = constant_size;
  RegisterHandleType(op->buffer_var.get(), op->type);
  /* TODO : Intel does not support array partitioning
  for (size_t i = 0; i < op->attrs.size(); i++) {
    this->PrintStmt(op->attrs[i]);
  }*/
  this->PrintStmt(op->body);
}

void CodeGenIntelHLS::VisitStmt_(const Partition* op) {
  stream << "#pragma HLS array_partition variable=";
  std::string vid = GetVarID(op->buffer_var.get());
  stream << vid << " ";
  switch (op->partition_type) {
    case PartitionType::Complete:
      stream << "complete";
      break;
    case PartitionType::Block:
      stream << "block";
      break;
    case PartitionType::Cyclic:
      stream << "cyclic";
      break;
  }
  stream << " dim=" << op->dim;
  if (op->partition_type != PartitionType::Complete) {
    stream << " factor=" << op->factor;
  }
  stream << "\n";
}

}  // namespace codegen
}  // namespace TVM
