/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen_c.h
 * \brief Common utilities to generated C style code.
 */
#ifndef CODEGEN_CODEGEN_C_H_
#define CODEGEN_CODEGEN_C_H_

#include <tvm/codegen.h>
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_visitor.h>
#include <tvm/lowered_func.h>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../runtime/thread_storage_scope.h"
#include "./codegen_source_base.h"
#include "./merlinc/codeanalys_merlinc.h"

namespace TVM {
namespace codegen {

template <class T, class V>
using str2tupleMap = std::unordered_map<std::string, std::tuple<T, V>>;

Type String2Type(std::string& s);
std::string getIndex(std::vector<int> shape);

/*!
 * \brief A base class to generate C code.
 *
 *  CodeGenC have two modes: generate SSA formed C code or normal form.
 *
 * **NOTE** CodeGenC does not aim at generating C codes consumed by MSVC or GCC,
 * Rather, it's providing infrastructural abstraction for C variants like CUDA
 * and OpenCL-C. You might find some odd variant features, e.g., type `int3` for
 * a vector of 3 `int`s. For native C code generator, see `CodeGenLLVM`.
 */
class CodeGenC : public ExprFunctor<void(const Expr&, std::ostream&)>,
                 public StmtFunctor<void(const Stmt&)>,
                 public CodeGenSourceBase {
 public:
  /*!
   * \brief Initialize the code generator.
   * \param output_ssa Whether output SSA.
   */
  void Init(bool output_ssa);
  /*!
   * \brief Add the function to the generated module.
   * \param f The function to be compiled.
   */
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  std::string Finish();
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The host code.
   */
  std::string GetHost();
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The device code.
   */
  std::string GetDevice();
  std::string GetConfig();
  /*!
   * \brief Print the Stmt n to CodeGenC->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n) { VisitStmt(n); }
  /*!
   * \brief Print the expression n(or its ssa id if in ssa mode) into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr(const Expr& n, std::ostream& os);
  /*!
   * \brief Same as PrintExpr, but simply returns result string
   * \param n The expression to be printed.
   */
  std::string PrintExpr(const Expr& n) {
    std::ostringstream os;
    PrintExpr(n, os);
    return os.str();
  }
  // The following parts are overloadable print operations.
  /*!
   * \brief Initialize codegen state for generating f.
   * \param f The function to be compiled.
   */
  virtual void InitFuncState(LoweredFunc f);
  // expression
  void VisitExpr_(const Variable* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const Load* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const Let* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Call* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const Add* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Sub* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Mul* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Div* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Mod* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Min* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Max* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const EQ* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const NE* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const LT* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const LE* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const GT* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const GE* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const And* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Or* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const Cast* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const Not* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const Select* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const Ramp* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const Broadcast* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const IntImm* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const UIntImm* op, std::ostream& os) override;    // NOLINT(*)
  void VisitExpr_(const FloatImm* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const StringImm* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const GetBit* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const GetSlice* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const SetBit* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const SetSlice* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const Quantize* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const KernelExpr* op,
                  std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const StreamExpr* op,
                  std::ostream& os) override;  // NOLINT(*)
  // statment
  void VisitStmt_(const LetStmt* op) override;
  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const Stencil* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const AttrStmt* op) override;
  void VisitStmt_(const ExternModule* op) override;
  void VisitStmt_(const AssertStmt* op) override;
  void VisitStmt_(const Evaluate* op) override;
  void VisitStmt_(const Block* op) override;
  void VisitStmt_(const ProducerConsumer* op) override;
  void VisitStmt_(const KernelDef* op) override;
  void VisitStmt_(const KernelStmt* op) override;
  void VisitStmt_(const StreamStmt* op) override;
  void VisitStmt_(const Return* op) override;
  void VisitStmt_(const Break* op) override;
  void VisitStmt_(const While* op) override;
  void VisitStmt_(const Partition* op) override;
  void VisitStmt_(const Print* op) override;
  /*!
   * Print Type represetnation of type t.
   * \param t The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(Type t, std::ostream& os);  // NOLINT(*)
  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex(const IterVar& iv);  // NOLINT(*)
  virtual void PrintStorageScope(const std::string& scope,
                                 std::ostream& os);  // NOLINT(*)
  virtual void PrintStorageSync(const Call* op);     // NOLINT(*)
  // Binary vector op.
  virtual void PrintVecBinaryOp(const std::string& op, Type op_type, Expr lhs,
                                Expr rhs, std::ostream& os);  // NOLINT(*)
  // print vector load
  virtual std::string GetVecLoad(Type t, const Variable* buffer, Expr base);
  // print vector store
  virtual void PrintVecStore(const Variable* buffer, Type t, Expr base,
                             const std::string& value);  // NOLINT(*)
  // print load of single element
  virtual void PrintVecElemLoad(const std::string& vec, Type t, int i,
                                std::ostream& os);  // NOLINT(*)
  // print store of single element.
  virtual void PrintVecElemStore(const std::string& vec, Type t, int i,
                                 const std::string& value);
  // get a cast type from to
  virtual std::string CastFromTo(std::string value, Type from, Type target);

  // map from var to shape, range and type
  std::map<const Variable*, Array<Expr>> var_shape_map_;
  std::unordered_map<const Variable*, Expr> range_;
  str2tupleMap<std::string, Type> map_arg_type_;
  // allocated buffer names
  std::unordered_set<std::string> alloc_set_;

  // save for kernel
  std::map<const Variable*, Array<Expr>> var_shape_map_save;
  std::unordered_map<const Variable*, Expr> range_save;
  std::unordered_set<std::string> alloc_set_save;

  // top function argument names
  std::vector<std::string> arg_names;

 protected:
  void SaveFuncState(LoweredFunc f);
  void RestoreFuncState(LoweredFunc f);
  void PrintArray(const Array<Expr>& array, const std::vector<size_t>& extents,
                  std::ostringstream& stream, size_t offset, size_t level);
  bool PrintConstants(const Stmt& stmt, bool multi_dim);
  class ConstantsPrinter;

  // Print reference to struct location
  std::string GetStructRef(Type t, const Expr& buffer, const Expr& index,
                           int kind);
  // print reference to a buffer as type t in index.
  virtual std::string GetBufferRef(Type t, const Variable* buffer, Expr index);
  /*!
   * \brief If buffer is allocated as type t.
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  bool HandleTypeMatch(const Variable* buf_var, Type t) const;
  /*!
   * \brief Register the data type of buf_var
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  void RegisterHandleType(const Variable* buf_var, Type t);
  // override
  void PrintSSAAssign(const std::string& target, const std::string& src,
                      Type t) final;
  /*! \brief restrict keyword */
  std::string restrict_keyword_{""};
  /*! \brief the custom compiler flags*/
  std::ostringstream cfg_stream;
  /*! \brief the storage scope of allocation */
  std::unordered_map<const Variable*, std::string> alloc_storage_scope_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const Variable*, Type> handle_data_type_;
  std::unordered_map<const Variable*, int> buf_length_map_;

  // save for kernel generation
  std::unordered_map<const Variable*, std::string> alloc_storage_scope_save;
  std::unordered_map<const Variable*, Type> handle_data_type_save;
  std::unordered_map<const Variable*, std::string> var_idmap_save;
  std::unordered_map<std::string, int> name_alloc_map_save;
  std::unordered_map<std::string, SSAEntry> ssa_assign_map_save;
  std::vector<bool> scope_mark_save;

 private:
  /*! \brief whether to print in SSA form */
  bool print_ssa_form_{false};
  /*! \brief set of volatile buf access */
  std::unordered_set<const Variable*> volatile_buf_;
};

}  // namespace codegen
}  // namespace TVM
#endif  // CODEGEN_CODEGEN_C_H_
