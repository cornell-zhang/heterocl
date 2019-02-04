/*!
 *  Copyright (c) 2019 by Contributors
 * \file compute_primitive.h
 */
#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include "./graph.h"

namespace tvm {

Stmt SplitLoop(Stmt& stmt,
               const IterVar& parent,
               const Expr factor,
               const Expr nparts,
               const IterVar& outer,
               const IterVar& inner);

Stmt FuseLoop(Stmt& stmt,
              const IterVar& outer,
              const IterVar& inner,
              const IterVar& fused);

Stmt ReorderLoop(Stmt& stmt, const Array<IterVar>& order);

Stmt PerformComputeAt(Stmt& producer,
                      Stmt& consumer,
                      const IterVar& var,
                      size_t& attach_level);

Stmt UpdateIterVarAttr(Stmt& stmt,
                      const IterVar& var,
                      const IterVarAttrNode* node);

} // namespace tvm
