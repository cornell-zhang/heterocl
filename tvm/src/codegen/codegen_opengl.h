/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_opengl.h
 * \brief Generate OpenGL device code.
 */
#ifndef CODEGEN_CODEGEN_OPENGL_H_
#define CODEGEN_CODEGEN_OPENGL_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "../runtime/opengl/opengl_module.h"
#include "./codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenOpenGL final : public CodeGenC {
 public:
  CodeGenOpenGL();
  void AddFunction(LoweredFunc f);
  std::unordered_map<std::string, runtime::OpenGLShader> Finish();

  void InitFuncState(LoweredFunc f) final;
  void BindThreadIndex(const IterVar& iv) final;
  void VisitStmt_(const Store* op) final;
  std::string TexelFetch(const Variable* buffer, Expr index);
  std::string GetBufferRef(Type t, const Variable* buffer, Expr index) final;
  void PrintType(Type t, std::ostream& os) final;  // NOLINT(*)

  // Codegen for immediate values
  void VisitExpr_(const IntImm* op, std::ostream& os) final;     // NOLINT(*)
  void VisitExpr_(const UIntImm* op, std::ostream& os) final;    // NOLINT(*)
  void VisitExpr_(const FloatImm* op, std::ostream& os) final;   // NOLINT(*)
  void VisitExpr_(const StringImm* op, std::ostream& os) final;  // NOLINT(*)

  // Match glsl_texture_store Call.
  void VisitStmt_(const Evaluate* op) final;  // NOLINT(*)

 private:
  const Variable* output_{nullptr};
  std::unordered_set<const Variable*> inputs_;
  const Variable* output_iter_var_{nullptr};
  std::unordered_map<std::string, runtime::OpenGLShader> shaders_;
  std::string thread_extent_var_;
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_CODEGEN_OPENGL_H_
