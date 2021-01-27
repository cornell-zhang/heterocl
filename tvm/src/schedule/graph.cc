/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.cc
 * \brief Utilities to get information about schedule graph.
 */
#include "graph.h"
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <unordered_map>
#include <unordered_set>

namespace TVM {
namespace schedule {
// key to specific tensor dimension.
struct TensorDimKey {
  FunctionRef f;
  int value_index;
  int dim;
  TensorDimKey() {}
  TensorDimKey(const ir::Call* op, int dim)
      : f(op->func), value_index(op->value_index), dim(dim) {}
  TensorDimKey(const Tensor& t, int dim)
      : f(t->op), value_index(t->value_index), dim(dim) {}
  TensorDimKey(const Tensor& t, size_t dim)
      : f(t->op), value_index(t->value_index), dim(static_cast<int>(dim)) {}
  inline bool operator==(const TensorDimKey& other) const {
    return f == other.f && value_index == other.value_index && dim == other.dim;
  }
  inline bool operator!=(const TensorDimKey& other) const {
    return !operator==(other);
  }
};
}  // namespace schedule
}  // namespace TVM

namespace std {
template <>
struct hash<::TVM::schedule::TensorDimKey> {
  std::size_t operator()(const ::TVM::schedule::TensorDimKey& k) const {
    size_t lhs = k.f.hash();
    size_t rhs =
        static_cast<size_t>(k.value_index) << 16UL | static_cast<size_t>(k.dim);
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};
}  // namespace std

namespace TVM {
namespace schedule {

// construct a read graph that gives readers of each operation
// that the root depend on
ReadGraph CreateReadGraph(const Array<Operation>& roots, const Schedule& sch) {
  ReadGraph rmap;
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  // initialize the roots
  for (Operation op : roots) {
    stack.push_back(op);
    visited.insert(op.get());
  }

  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
    Array<Tensor> deps = op->InputTensors();
    Array<Tensor> new_deps;
    for (Tensor t : deps) {
      if (t->op.defined()) {
        // need to get the updated op
        Operation dep_op;
        if (sch->stage_map.size() == 0) {
          dep_op = t->op;
          new_deps.push_back(t);
        } else {
          dep_op = sch->stage_map[t->op]->op;
          new_deps.push_back(dep_op.output(0));
        }
        if (visited.count(dep_op.get()) == 0) {
          visited.insert(dep_op.get());
          stack.push_back(dep_op);
        }
      }
    }
    rmap.Set(op, new_deps);
  }
  return rmap;
}

// Do DFS visit to get the subgraph.
// Return if op is inside the subgraph.
bool GetSubGraphByPostDFS_(const Operation& op,
                           const std::unordered_set<const Node*>& boundary,
                           bool include_bounary,
                           std::unordered_map<const Node*, bool>* visited,
                           Array<Operation>* result) {
  if (visited->count(op.get())) {
    return visited->at(op.get());
  }
  if (boundary.count(op.get())) {
    (*visited)[op.get()] = true;
    if (include_bounary) {
      result->push_back(op);
    }
    return true;
  }
  // mark to avoid loop
  // Not necessary for DAG.
  (*visited)[op.get()] = false;
  // check if we can reach boundary.
  bool reach_boundary = false;
  for (Tensor t : op->InputTensors()) {
    if (GetSubGraphByPostDFS_(t->op, boundary, include_bounary, visited,
                              result)) {
      reach_boundary = true;
    }
  }
  (*visited)[op.get()] = reach_boundary;
  if (reach_boundary) {
    result->push_back(op);
  }
  return reach_boundary;
}

Array<Operation> GetSubGraph(const Array<Tensor>& outputs,
                             const Array<Tensor>& inputs, bool include_inputs) {
  Array<Operation> result;
  std::unordered_set<const Node*> boundary;
  for (Tensor t : inputs) {
    boundary.insert(t->op.get());
  }
  std::unordered_map<const Node*, bool> visited;
  for (Tensor t : outputs) {
    GetSubGraphByPostDFS_(t->op, boundary, include_inputs, &visited, &result);
  }
  return result;
}

void PostDFSOrder(const Operation& op, const ReadGraph& g,
                  std::unordered_set<Operation>* visited,
                  Array<Operation>* post_order) {
  if (visited->count(op)) return;
  visited->insert(op);
  for (const auto& t : g.at(op)) {
    PostDFSOrder(t->op, g, visited, post_order);
  }
  post_order->push_back(op);
}

Array<Operation> PostDFSOrder(const Array<Operation>& roots,
                              const ReadGraph& g) {
  std::unordered_set<Operation> visited;
  Array<Operation> post_order;
  for (Operation op : roots) {
    PostDFSOrder(op, g, &visited, &post_order);
  }
  return post_order;
}

FeedGraph CreateFeedGraph(const ReadGraph& g) {
  FeedGraph fg;
  for (auto kv : g) {
    for (Tensor t : kv.second) {
      fg[t].push_back(kv.first);
    }
  }
  return fg;
}

AttachPath CreateAttachPath(Schedule sch) {
  AttachPath ret;
  for (Stage stage : sch->stages) {
    std::unordered_set<const Node*> visited;
    Array<IterVar> path;
    for (Stage s = stage; s.defined();) {
      CHECK(!visited.count(s.get())) << "Find loop in compute_at attach group";
      visited.insert(s.get());
      Stage spec = s.GetAttachSpec();
      bool start_attach;
      IterVar attach_ivar;
      if (spec->attach_type == kScope) {
        attach_ivar = spec->attach_ivar;
        s = spec->attach_stage;
        start_attach = false;
        CHECK(attach_ivar.defined());
      } else {
        break;
      }
      CHECK(s.defined());
      for (size_t i = s->leaf_iter_vars.size(); i != 0; --i) {
        IterVar iv = s->leaf_iter_vars[i - 1];
        if (!start_attach && iv.same_as(attach_ivar)) {
          start_attach = true;
        }
        if (start_attach) path.push_back(iv);
      }
      CHECK(start_attach) << "Invalid Schedule: cannot find attach point "
                          << attach_ivar << " in the schedule of " << s->op;
    }
    if (!ret.count(stage->op)) {
      ret.Set(stage->op, path);
    }
  }
  return ret;
}

// graph of push reach relation of tensor dimensions
using ReachGraph = std::unordered_map<TensorDimKey, std::vector<TensorDimKey>>;

ReachGraph GetReachGraph(const Array<Operation>& ops) {
  ReachGraph reach;
  std::unordered_set<const Node*> bset;
  for (size_t i = 0; i < ops.size(); ++i) {
    bset.insert(ops[i].get());
  }
  return reach;
}

}  // namespace schedule
}  // namespace TVM
