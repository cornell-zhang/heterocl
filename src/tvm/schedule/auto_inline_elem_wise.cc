/*!
 *  Copyright (c) 2016 by Contributors
 * \file auto_inline_elem_wise.cc
 */
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>

namespace TVM {
namespace schedule {

using namespace ir;

class ElemWiseDetector : public ir::IRVisitor {
 public:
  explicit ElemWiseDetector(Array<IterVar> axis) : axis_(axis) {}

  void Visit(const NodeRef& e) final {
    if (!is_elem_wise_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Call* op) final {
    Array<Expr> axis = op->args;
    if (axis_.size() != axis.size()) {
      is_elem_wise_ = false;
      return;
    }

    for (size_t i = 0; i < axis_.size(); ++i) {
      if (!axis[i].same_as(axis_[i]->var)) {
        is_elem_wise_ = false;
        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  bool is_elem_wise_{true};

 private:
  Array<IterVar> axis_;
};

bool IsElemWise(const Operation& op) { return false; }

void AutoInlineElemWise(Schedule sch) {
  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && IsElemWise(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

bool IsBroadcast(const Operation& op) { return false; }

void AutoInlineBroadcast(Schedule sch) {
  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && IsBroadcast(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

bool IsInjective(const Operation& op) { return false; }

void AutoInlineInjective(Schedule sch) {
  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && IsInjective(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

}  // namespace schedule
}  // namespace TVM
