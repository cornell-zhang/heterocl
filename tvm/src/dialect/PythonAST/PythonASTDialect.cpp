//===- PythonASTDialect.cpp - PythonAST dialect ---------------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#include "dialect/PythonAST/PythonASTDialect.h"
#include "dialect/PythonAST/PythonASTOps.h"

using namespace mlir;
using namespace mlir::pythonast;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void PythonASTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialect/PythonAST/PythonASTOps.cpp.inc"
      >();
}
