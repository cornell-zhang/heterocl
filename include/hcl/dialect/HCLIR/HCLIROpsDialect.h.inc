/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace hclir {

class HCLIRDialect : public ::mlir::Dialect {
  explicit HCLIRDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<HCLIRDialect>()) {
    
    initialize();
  }
  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("HCLIR");
  }
};
} // namespace hclir
} // namespace mlir
