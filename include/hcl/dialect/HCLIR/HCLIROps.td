//===- HCLIROps.td -----------------------*- tablegen -*-===//
//  Copyright (c) 2021 by Contributors
//
//  Operations for HeteroCL IR
//===----------------------------------------------------===//

#ifndef HCL_DIALECT_HCLIR_OPS
#define HCL_DIALECT_HCLIR_OPS

include "HCLIR/HCLIRDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AddOp : HCLIR_Op<"add"> {

    let summary = "Add operator";
    let description = [{
        Add two operand
    }];

    let arguments = (ins AnyType:$a, AnyType:$b);
    let results = (outs AnyType);
}

#endif // HCL_DIALECT_HCLIR_OPS
