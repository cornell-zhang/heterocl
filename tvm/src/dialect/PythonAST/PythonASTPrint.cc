/*!
 *  Copyright (c) 2021 by Contributors
 * \file PythonAST.cc
 */
#include <dialect/PythonAST/PythonAST.h>

namespace TVM {
namespace AST {

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<LocationNode>([](const LocationNode *op, IRPrinter *p) {
      #ifdef PRINT_LOC
      p->stream << "(" << op->file_name << ": "
                << op->line << ":"
                << op->column << ")";
      #endif
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Var>([](const Var *op, IRPrinter *p) {
      p->stream << op->name;
      p->print(op->loc);
    });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<Add>([](const Add *op, IRPrinter *p) {
      p->print(op->lhs);
      p->stream << "+";
      p->print(op->loc);
      p->print(op->rhs);
    });

}  // namespace AST
}  // namespace TVM
