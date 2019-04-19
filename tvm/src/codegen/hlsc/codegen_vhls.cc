/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "./codegen_vhls.h"
#include "../build_common.h"
#include "../codegen_soda.h"
#include "../../pass/stencil.h"

namespace tvm {
namespace codegen {

void CodeGenVivadoHLS::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // Write header files
  this->stream << "#include <ap_int.h>\n";
  this->stream << "#include <ap_fixed.h>\n";
  this->stream << "#include <math.h>\n\n";
  CodeGenHLSC::AddFunction(f, map_arg_type);
}

void CodeGenVivadoHLS::PrintType(Type t, std::ostream& os) {
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.bits() == 32 && t.fracs() == 0) {
      if (t.is_uint()) os << "unsigned ";
      os << "int";
    }
    else if (t.is_uint())   os << "ap_uint<" << t.bits() << ">";
    else if (t.is_int())    os << "ap_int<" << t.bits() << ">";
    else if (t.is_ufixed()) os << "ap_ufixed<" << t.bits() << ", " << t.fracs() << ">";
    else                    os << "ap_fixed<" << t.bits() << ", " << t.fracs() << ">";
  } else {
    CodeGenC::PrintType(t, os);
  }
}

void CodeGenVivadoHLS::VisitExpr_(const GetBit* op, std::ostream& os) {
  PrintExpr(op->a, os);
  os << "[";
  PrintExpr(op->index, os);
  os << "]";
}

void CodeGenVivadoHLS::VisitExpr_(const GetSlice* op, std::ostream& os) {
  PrintExpr(op->a, os);
  os << "(";
  PrintExpr(op->index_left, os);
  os << ", ";
  PrintExpr(op->index_right, os);
  os << ")";
}

void CodeGenVivadoHLS::VisitStmt_(const Store* op) {
  // handle SetSlice
  if (const SetSlice* ss = op->value.as<SetSlice>()) {
    Type t = op->value.type();
    Expr new_index_left = ir::Simplify(ss->index_left - 1);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent(); 
    this->stream << ref
                 << "(" << PrintExpr(new_index_left) << ", " << PrintExpr(ss->index_right)
                 << ") = " << PrintExpr(ss->value) << ";\n";
  } else if (const SetBit* sb = op->value.as<SetBit>()) {
    Type t = op->value.type();
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent();
    this->stream << ref
                 << "[" << PrintExpr(sb->index)
                 << "] = " << PrintExpr(sb->value) << ";\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenVivadoHLS::VisitStmt_(const For* op) {
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

void CodeGenVivadoHLS::VisitStmt_(const Allocate* op) {
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
  stream << ' '<< vid;
  if (constant_size > 1) {// Transfer length one array to scalar
    for (size_t i = 0; i < op->extents.size(); i++) {
      stream << '[';
      PrintExpr(op->extents[i], stream);
      stream << "]";
    }
  }
  stream << ";\n";
  buf_length_map_[buffer] = constant_size;
  RegisterHandleType(op->buffer_var.get(), op->type);
  for (size_t i = 0; i < op->attrs.size(); i++) {
    this->PrintStmt(op->attrs[i]);
  }
  this->PrintStmt(op->body);
}

void CodeGenVivadoHLS::VisitStmt_(const Partition* op) {
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

void CodeGenVivadoHLS::VisitStmt_(const Stencil* op) {
  CodeGenSODA cg_soda;
  cg_soda.Init(false);
  VarExprUnorderedSet inputs;
  VarExprUnorderedSet outputs;
  for (size_t i = 0; i < op->inputs.size(); i++)
    inputs.insert(op->inputs[i]);
  for (size_t i = 0; i < op->outputs.size(); i++) {
    outputs.insert(op->outputs[i]);
    LOG(INFO) << op->outputs[i].get();
  }
  std::string func_name = "soda_" + 
                          op->inputs[0]->name_hint + "_" +
                          op->outputs[0]->name_hint;
  cg_soda.PrintSODA(func_name, op->burst_width, op->unroll_factor,
      op->num_iteration, op->body, inputs, outputs);
  std::string code = cg_soda.Finish();
  LOG(INFO) << code;

  // writh SODA to a separate file
  // TODO: create a function that reuses the following part
  // Mangle PATH to find sodac
  if (char* pythonpath = getenv("PYTHONPATH")) {
    char* path = strtok(pythonpath, ":");
    while (path != nullptr) {
      setenv("PATH",
          (std::string(path) + "/../soda/src:" + getenv("PATH")).c_str(),
          /* overwrite = */1);
      path = strtok(nullptr, ":");
    }
  }

  // Check that python3 and sodac are there
  if (system("which python3 >/dev/null") != 0) {
    LOG(WARNING) << "python3 not found";
  }
  if (system("which sodac >/dev/null") != 0) {
    LOG(WARNING) << "sodac not found";
  }

  // Invoke sodac
  auto check = [](int returned, int expected = 0) {
    if (returned != expected) {
      LOG(WARNING) << strerror(errno);
      exit(errno);
    }
  };

  // Create pipes for inter-process communication
  int pipe0[2];
  int pipe1[2];
  int pipe2[2];
  check(pipe(pipe0));
  check(pipe(pipe1));
  check(pipe(pipe2));

  // Fork to prepare for inter-process communication
  pid_t pid = fork();
  if (pid == -1) { LOG(WARNING) << strerror(errno); }
  if (pid) {  // Parent process
    // Close unused read end of pipe0 and write ends of pipe1 & pipe2
    check(close(pipe0[0]));
    check(close(pipe1[1]));
    check(close(pipe2[1]));

    // Write SODA DSL to the write end of pipe0
    check(write(pipe0[1], code.c_str(), code.size()), code.size());

    // Close write end of pipe0 to generate EOF
    check(close(pipe0[1]));

    // Open the read ends of pipe1 & pipe2
    std::ifstream stream1("/proc/self/fd/" + std::to_string(pipe1[0]));
    std::ifstream stream2("/proc/self/fd/" + std::to_string(pipe2[0]));

    // Close the old fds of the read ends of pipe1 & pipe2
    check(close(pipe1[0]));
    check(close(pipe2[0]));

    // Read pipe1 & pipe2
    using InputIter = std::istreambuf_iterator<char>;
    std::string content1((InputIter(stream1)), InputIter());
    std::string content2((InputIter(stream2)), InputIter());

    // Use child's stdout as the code output
    code = content1;

    // Use child's stderr as logging messages
    if (!content2.empty()) {
      LOG(INFO) << content2;
    }

    wait(nullptr);
  } else {  // Child process
    // Close unused write end of pipe0 and read ends of pipe1 & pipe2
    check(close(pipe0[1]));
    check(close(pipe1[0]));
    check(close(pipe2[0]));

    // Replace stdin, stdout, and stderr with pipe0, pipe1, and pipe2
    check(dup2(pipe0[0], 0), 0);
    check(dup2(pipe1[1], 1), 1);
    check(dup2(pipe2[1], 2), 2);

    // Close old fds of pipe0, pipe1, and pipe2
    check(close(pipe0[0]));
    check(close(pipe1[1]));
    check(close(pipe2[1]));

    // Invoke sodac
    check(execlp("/bin/sh", "/bin/sh", "-c",
          "python3 $(which sodac) --xocl-kernel - -", nullptr));
  }

  stream << code;
}

}  // namespace codegen
}  // namespace tvm
