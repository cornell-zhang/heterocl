//===- HCLIRTDialect.td ------------------*- tablegen -*-===//
//  Copyright (c) 2021 by Contributors
//
//  Dialect for HeteroCL IR
//===----------------------------------------------------===//

#ifndef HCL_DIALECT_HCLIR_DIALECT
#define HCL_DIALECT_HCLIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// HCL IR dialect definition.
//===----------------------------------------------------------------------===//

def HCLIR_Dialect : Dialect {
    let name = "HCLIR";
    let summary = "A HeteroCL dialect.";
    let description = [{
        A HeteroCL dialect.
    }];
    let cppNamespace = "::mlir::hclir";
}

//===----------------------------------------------------------------------===//
// Base HCL IR operation definition.
//===----------------------------------------------------------------------===//

class HCLIR_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<HCLIR_Dialect, mnemonic, traits>;

#endif // HCL_DIALECT_HCLIR_DIALECT
