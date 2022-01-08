/*!
 *  Copyright (c) 2021 by Contributors
 * \file code_analysis.h
 * \brief AST analyzer for arguement types.
 */
#ifndef CODEGEN_CODE_ANALYSIS_H_
#define CODEGEN_CODE_ANALYSIS_H_

#include <tvm/codegen.h>
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/lowered_func.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "./codegen_source_base.h"

namespace TVM {
namespace codegen {

using namespace ir;

template <class T, class V>
using str2tupleMap = std::unordered_map<std::string, std::tuple<T, V>>;

/*!
 * \brief A class to analyze the IR AST for MerlinC generation.
 *
 */
class CodeAnalysis : public ExprFunctor<void(const Expr&, std::ostream&)>,
                     public StmtFunctor<void(const Stmt&)>,
                     public CodeGenSourceBase {
 public:
  /*!
   * \brief Initialize the code generator.
   * \param output_ssa Whether output SSA.
   */
  void Init();
  /*!
   * \brief Add the function to the generated module.
   * \param f The function to be compiled.
   */
  void AddFunction(LoweredFunc f);
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  str2tupleMap<std::string, Type> Finish();
  /*!
   * \brief Print the Stmt n to CodeAnalysis->stream
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
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const AttrStmt* op) override;
  void VisitStmt_(const AssertStmt* op) override;
  void VisitStmt_(const Evaluate* op) override;
  void VisitStmt_(const Block* op) override;
  void VisitStmt_(const ProducerConsumer* op) override;
  void VisitStmt_(const KernelDef* op) override;
  void VisitStmt_(const KernelStmt* op) override;
  void VisitStmt_(const Return* op) override;
  void VisitStmt_(const Break* op) override;
  void VisitStmt_(const While* op) override;
  void VisitStmt_(const Reuse* op) override;
  void VisitStmt_(const Partition* op) override;
  void VisitStmt_(const Stencil* op) override;
  void VisitStmt_(const ExternModule* op) override;
  void VisitStmt_(const StreamStmt* op) override;
  void VisitStmt_(const Print* op) override;
  /*!
   * Print Type represetnation of type t.
   * \param t The type representation.
   * \param os The stream to print the ctype into
   */
  void PrintType(Type t, std::ostream& os);  // NOLINT(*)
  std::string GetType(Type t);               // NOLINT(*)
  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  void BindThreadIndex(const IterVar& iv);  // NOLINT(*)
  void PrintStorageScope(const std::string& scope,
                         std::ostream& os);  // NOLINT(*)
  void PrintStorageSync(const Call* op);     // NOLINT(*)
  // Binary vector op.
  void PrintVecBinaryOp(const std::string& op, Type op_type, Expr lhs, Expr rhs,
                        std::ostream& os);  // NOLINT(*)
  // print vector load
  std::string GetVecLoad(Type t, const Variable* buffer, Expr base);
  // print vector store
  void PrintVecStore(const Variable* buffer, Type t, Expr base,
                     const std::string& value);  // NOLINT(*)
  // print load of single element
  void PrintVecElemLoad(const std::string& vec, Type t, int i,
                        std::ostream& os);  // NOLINT(*)
  // print store of single element.
  void PrintVecElemStore(const std::string& vec, Type t, int i,
                         const std::string& value);
  // Get a cast type from to
  std::string CastFromTo(std::string value, Type from, Type target);

 protected:
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
  /*! \brief the storage scope of allocation */
  std::unordered_map<const Variable*, std::string> alloc_storage_scope_;
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const Variable*, Type> handle_data_type_;

 private:
  /*! \brief set of volatile buf access */
  std::unordered_set<const Variable*> volatile_buf_;
  /*! \brief map of function arguments to their types */
  str2tupleMap<std::string, Type> map_arg_type_;
};

}  // namespace codegen
}  // namespace TVM
#endif  // CODEGEN_CODE_ANALYSIS_H_
