//===- PythonASTDialect.td - PythonAST dialect -----------*- tablegen -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_PYTHONAST_DIALECT
#define DIALECT_PYTHONAST_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PythonAST dialect definition.
//===----------------------------------------------------------------------===//

def PythonAST_Dialect : Dialect {
    let name = "python_ast";
    let summary = "A dialect for frontend Python AST.";
    let description = [{
        A Python AST dialect.
    }];
    let cppNamespace = "::mlir::pythonast";
}

//===----------------------------------------------------------------------===//
// Base PythonAST operation definition.
//===----------------------------------------------------------------------===//

class PythonAST_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<PythonAST_Dialect, mnemonic, traits>;

#endif // DIALECT_PYTHONAST_DIALECT
