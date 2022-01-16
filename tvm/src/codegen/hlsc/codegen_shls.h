/*!
 * Copyright (c) 2021 by Contributors
 * \file codegen_vhls.h
 * \brief Generate Stratus HLS kernel code.
 */
#ifndef CODEGEN_HLSC_CODEGEN_SHLS_H_
#define CODEGEN_HLSC_CODEGEN_SHLS_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <fstream>
#include <string>
#include "./codegen_hlsc.h"
#include "./codegen_vhls.h"

namespace TVM {
namespace codegen {

class CodeGenStratusHLS final : public CodeGenVivadoHLS {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os);
  void PrintType(Type t, std::ostream& os, bool is_index);

  // Stmt Printing
  void VisitStmt_(const For* op);
  void VisitStmt_(const Store* op);
  void VisitStmt_(const Allocate* op);
  void VisitStmt_(const Partition* op);
  void VisitStmt_(const KernelDef* op);
  void VisitStmt_(const KernelStmt* op);
  void VisitStmt_(const Return* op);
  void VisitStmt_(const Assert* op) final;

  // Expr Printing
  void VisitExpr_(const Load* op, std::ostream& os);
  void VisitExpr_(const KernelExpr* op, std::ostream& os);
  void VisitExpr_(const SetBit* op, std::ostream& os);
  void VisitExpr_(const SetSlice* op, std::ostream& os);
  void VisitExpr_(const Cast* op, std::ostream& os);
  void VisitExpr_(const IntImm* op, std::ostream& os);
  void VisitExpr_(const UIntImm* op, std::ostream& os);

  // Finish
  std::string Finish();
  std::string GetHost();
  std::string GetDevice();

  // Misc
  void printTclFile();
  std::string GetBufferRef(Type t, const Variable* buffer, Expr index);
  void GenForStmt(const For* op, std::string pragma, bool before);
  void PrintTypeStringImm(const StringImm* t, std::ostream& os);
  bool IsP2P(const std::string& vid);
  void UpdateSubPortTypes(
      const std::map<std::string, std::vector<Expr>> arg_types);
  void UpdateSubPortNames(
      const std::map<std::string, std::vector<std::string>> arg_names);
  std::string CastFromTo(std::string value, Type from, Type target);
  void GenerateModule(std::string name, bool top_level,
                      str2tupleMap<std::string, Type> map_arg_type,
                      const Array<Expr> arg_types,
                      const Array<Array<Expr>> arg_shapes, const Stmt body,
                      const Array<VarExpr> args, std::ostringstream& decl_os,
                      std::ostringstream& ctor_os, std::ostringstream& body_os);

  // Formatting
  void PrintIndentHeader();
  void PrintIndentCtor();
  int BeginScopeHeader();
  void EndScopeHeader(int scope_id);
  int BeginScopeCtor();
  void EndScopeCtor(int scope_id);
  void PrintIndentAnyStream(std::ostream& os, int indent);

 private:
  std::string _top_name{"dut"};
  std::map<std::string, std::string> _port_type;
  std::vector<std::vector<std::string>> _port_names;
  int level_{-1};    // module level
  int h_indent_{0};  // header indent
  int c_indent_{0};  // constructor indent
  std::vector<bool> h_scope_mark_;
  std::vector<bool> c_scope_mark_;
  std::vector<bool> scope_mark_;
  std::ostringstream ctor_stream;
  // submodules
  std::vector<std::string> sub_ctors;
  std::vector<std::string> sub_decls;
  std::vector<std::string> sub_threads;
  std::vector<std::string> sub_names;
  std::vector<std::string> support_files;
  std::vector<std::string> support_fnames;
  std::map<std::string, std::vector<std::string>> _sub_port_types;
  std::map<std::string, std::vector<std::string>> _sub_port_names;
  // external memory
  std::vector<std::string> ext_mem;
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_HLSC_CODEGEN_SHLS_H_
