/*!
 *  Copyright (c) 2020 by Contributors
 * \file schedule_reorder.cc
 * \brief The bound inference logic.
 */
#include <tvm/ir_visitor.h>
#include <tvm/schedule_pass.h>
#include <tvm/operation.h>
#include <tvm/ir_pass.h>
#include <unordered_map>
#include <unordered_set>
#include "./graph.h"
#include "./message_passing.h"
#include "../runtime/thread_storage_scope.h"

namespace TVM {
namespace schedule {

using namespace ir;

void ExtractSubGraph(const Array<Operation>& roots,
                     const ReadGraph& g,
                     std::unordered_map<const Node*, PlaceType>& dev,
                     std::unordered_set<const Node*>& boundary,
                     Array<Array<Tensor>>& inputs, Array<Array<Tensor>>& outputs) {
   
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  for (Operation op : roots) {
    stack.push_back(op);
    visited.insert(op.get());
  }

  while (!stack.empty() && boundary.size()) {
    Operation op = stack.back();
    stack.pop_back();
    LOG(INFO) << op;

    if (boundary.count(op.get())) {
      boundary.erase(boundary.find(op.get()));
    }

    for (const auto& t : g.at(op)) { 
      if (t->op.defined()) {
        if (visited.count(t->op.get()) == 0) {
          visited.insert(t->op.get());
          stack.push_back(t->op);
        }
      }
    }
  }
}

// schedule the ops 
void PostDFSSplit(const Operation& op,
                  const ReadGraph& g,
                  std::unordered_set<Operation>* visited,
                  Array<Operation>* post_order,
                  std::unordered_map<const Node*, PlaceType>& dev) {
  if (visited->count(op)) return;
  visited->insert(op);
  CHECK(dev.count(op.get())) << "not found " << op;

  // spread device scope from root to parents 
  Array<Tensor> tensors;
  if (dev[op.get()] != devHost) {
    for (const auto& t : g.at(op)) {
      if (dev[t->op.get()] == devHost) tensors.push_back(t);
    } 
    // propagate xcel scope to input tensors 
    if (tensors.size() == g.at(op).size()) { 
      for (const auto& t : g.at(op)) 
        dev[t->op.get()] = dev[op.get()];
    } else { // reorder inputs 
      for (const auto& t : g.at(op)) 
        if (dev[t->op.get()] != devHost) tensors.push_back(t);
      CHECK(tensors.size() == g.at(op).size());
    }
  }

  if (tensors.size() == 0) tensors = g.at(op);
  for (const auto& t : tensors) 
    PostDFSSplit(t->op, g, visited, post_order, dev);
  post_order->push_back(op);
}

// propagate device info thru op trees 
Array<Operation> PostDFSSplit(
    const Array<Operation>& roots,
    const ReadGraph& g, const Schedule& sch) {

  // map from op to stage device scope 
  std::unordered_map<const Node*, PlaceType> dev;
  std::unordered_set<const Node*> boundary;
  std::unordered_set<Operation> visited;
  for (Operation op : roots) dev[op.get()] = devHost;
  for (Stage stage : sch->stages) {
    if (dev.count(stage->op.get()))
      CHECK(dev[stage->op.get()] == devHost)
        << "output " << stage << " should be placed on host scope";
    dev[stage->op.get()] = stage->device_type;
    if (stage->device_type != devHost) 
      boundary.insert(stage->op.get());
  }
  
  // propagate device inforation  
  // the inputs and outputs marked with xcel scope indicators
  // are required to form an enclosed subgraph  
  Array<Array<Tensor>> inputs, outputs;
  ExtractSubGraph(roots, g, dev, boundary, inputs, outputs);

  Array<Operation> post_order;
  for (Operation op : roots) {
    PostDFSSplit(op, g, &visited, &post_order, dev);
  }
  return post_order;
}

// Infer the palcement for each stage. The placement  
// 1. Subgraphs will be repalced with a Stage of KernelStmt
// 2. Group stages in xcel scope and create KernelDef
Schedule ScopePartition(const Schedule& sch) {

  Array<Operation> roots;
  // std::unordered_map<Operation, Operation> opmap;
  for (Operation op : sch->outputs) { 
    roots.push_back(sch->stage_map[op]->op);
    // opmap[sch->stage_map[op]->op] = op;
  }

  // map from tensor to ops 
  ReadGraph rmap;
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  // create read graph from roots
  for (Operation op : roots) {
    // if(!sch->stage_map.count(op) && sch->stage_map.size() > 0) {
    //   for (Operation out_op : sch->outputs) { 
    //     if (out_op->name == op->name) {
    //       opmap[op] = out_op;
    //       CHECK(sch->stage_map.count(out_op));
    //     }
    //   }
    // }
    stack.push_back(op);
    visited.insert(op.get());
  }

  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
    Array<Tensor> deps = op->InputTensors();
    Array<Tensor> new_deps;
    for (Tensor t : deps) {
      // tensor as output of the operation 
      if (t->op.defined()) {
        Operation dep_op;
        CHECK(sch->stage_map.count(t->op))
          << "cannot find " << t->op;
        dep_op = sch->stage_map[t->op]->op;
        // if (dep_op.get() != t->op.get()) {
        //   opmap[dep_op] = t->op;
        // }
        new_deps.push_back(dep_op.output(0));
        // an tensor might be feeding multiple ops 
        if (visited.count(dep_op.get()) == 0) {
          visited.insert(dep_op.get());
          stack.push_back(dep_op);
        }
      }
    }
    rmap.Set(op, new_deps);
  }

  // map from op to stage
  std::unordered_map<const Node*, Stage> op2stage_;
  for (Stage stage : sch->stages) {
    std::shared_ptr<StageNode> snode =
        std::make_shared<StageNode>(*stage.operator->());
    op2stage_[stage->op.get()] = Stage(snode);
  }

  // for (auto& kv : rmap)
  //   LOG(INFO) << kv.first << ":" << kv.second;
   
  // re-sechdule ops array
  Array<Operation> post_order = PostDFSSplit(roots, rmap, sch);
  std::unordered_set<Operation> output_set;
  for (Operation x : sch->outputs) {
    output_set.insert(x);
  }

  // create new schedule node 
  std::shared_ptr<ScheduleNode> n = std::make_shared<ScheduleNode>();
  std::unordered_map<Stage, Stage, NodeHash, NodeEqual> smap;
  n->outputs = sch->outputs;

  // create new stages sharing same node 
  for (Operation op: post_order) {

    // hard fix: use op_map to find valid op
    // if(!sch->stage_map.count(op)) {
    //   LOG(INFO) << "missing op " << op;
    //   CHECK(opmap.count(op)) << "not found " << op;
    //   op = opmap[op];
    //   CHECK(sch->stage_map.count(op)) << "not found " << op;
    // }
    // Stage s = sch->stage_map[op]; 
    // Stage scopy = Stage(s.node_);

    CHECK(op2stage_.count(op.get()));
    // FIXME: inconsistent stage address
    // const Stage& s = op2stage_.at(op.get());

    Stage scopy;
    for (Stage s : sch->stages) {
      if (op->name == s->op->name) {
        std::shared_ptr<StageNode> snode =
            std::make_shared<StageNode>(*s.operator->());
        scopy = Stage(snode);
        smap[s] = scopy;
        n->stages.push_back(scopy);
      }
    }

    CHECK(scopy.defined());
    // FIXME: stage_map op inconsistent with s->op
    // CHECK(sch->stage_map.count(op));
  }

  // stages after dataflow rewrite
  // for (Stage s : sch->stages) {
  //   std::shared_ptr<StageNode> snode =
  //       std::make_shared<StageNode>(*s.operator->());
  //   Stage scopy = Stage(snode);
  //   smap[s] = scopy;
  //   n->stages.push_back(scopy);
  // }

  for (Stage g : sch->groups) {
    std::shared_ptr<StageNode> gnode =
        std::make_shared<StageNode>(*g.operator->());
    Stage gcopy = Stage(gnode);
    smap[g] = gcopy;
    n->groups.push_back(gcopy);
  }

  // remaps the reference relations.
  for (auto kv : sch->stage_map) { // set op to new stage
    CHECK(smap.count(kv.second)) << "not found " << kv.second;
    n->stage_map.Set(kv.first, smap.at(kv.second));
  }

  for (Stage s : n->stages) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
        << s->attach_stage << " not found";
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end())
        << s->group << " not found";
      s->group = smap.at(s->group);
    }
  }
  for (Stage s : n->groups) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
        << s->attach_stage << " not found";
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end())
        << s->group << " not found";
      s->group = smap.at(s->group);
    }
  }
  return Schedule(n);
}

}  // namespace schedule
}  // namespace TVM
