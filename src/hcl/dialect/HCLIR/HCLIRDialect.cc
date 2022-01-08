//===- HCLIRDialect.cc ----------------------*- C++ -*-===//
//  Copyright (c) 2021 by Contributors
//
//  Dialect for HeteroCL IR
//===--------------------------------------------------===//

#include "hcl/dialect/HCLIR/HCLIROps.h"
#include "hcl/dialect/HCLIR/HCLIRDialect.h"

using namespace mlir;
using namespace mlir::hclir;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void HCLIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hcl/dialect/HCLIR/HCLIROps.cpp.inc"
      >();
}
