//===- PythonASTOps.h - PythonAST dialect ops -----------------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_PYTHONAST_PYTHONASTOPS_H
#define DIALECT_PYTHONAST_PYTHONASTOPS_H

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "dialect/PythonAST/PythonASTOps.h.inc"

#endif // DIALECT_PYTHONAST_PYTHONASTOPS_H
