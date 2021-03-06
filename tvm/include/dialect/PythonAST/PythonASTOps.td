//===- PythonASTOps.td - PythonAST dialect ops -----------*- tablegen -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_PYTHONAST_OPS
#define DIALECT_PYTHONAST_OPS

include "PythonASTDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AddOp : PythonAST_Op<"add"> {

    let summary = "Add operator";
    let description = [{
        Add two operand
    }];

    let arguments = (ins AnyType:$a, AnyType:$b);
    let results = (outs AnyType);
}

#endif // DIALECT_PYTHONAST_OPS
