/*!
 *  Copyright (c) 2019 by Contributors
 * \file schedule_primitive.h
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

} // namespace tvm
