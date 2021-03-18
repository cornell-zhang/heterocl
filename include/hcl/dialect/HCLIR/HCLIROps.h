//===- HCLIROps.h ----------------------*- C++ -*-===//
//  Copyright (c) 2021 by Contributors
//
//  Operations for HeteroCL IR
//===---------------------------------------------===//

#ifndef HCL_DIALECT_HCLIR_HCLIROPS_H
#define HCL_DIALECT_HCLIR_HCLIROPS_H

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "hcl/dialect/HCLIR/HCLIROps.h.inc"

#endif // HCL_DIALECT_HCLIR_HCLIROPS_H
