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

std::unordered_set<Operation> ExtractAncestors(Operation root, const ReadGraph& g) {
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  std::unordered_set<Operation> ops;
  stack.push_back(root);
  visited.insert(root.get());
  // for (auto& kv : g) {
  //   LOG(INFO) << "------------";
  //   LOG(INFO) << kv.first;
  //   for (auto& t : kv.second) LOG(INFO) << t->op;
  // }

  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
    
    CHECK(g.count(op)) << "not found " << op;
    for (const auto& t : g.at(op)) { 
      if (t->op.defined()) {
        if (visited.count(t->op.get()) == 0) {
          ops.insert(t->op);
          visited.insert(t->op.get());
          stack.push_back(t->op);
        }
      }
    }
  }
  return ops;
}

Array<Tensor> RemapTensor(const Schedule& sch,
                          const Array<Tensor>& arr) {
  const auto& op2stage_cache = sch->op2stage_cache_;
  Array<Tensor> ret;
  for (Tensor t : arr) {
    if (!op2stage_cache.count(t->op.get())) {
      CHECK(sch->stage_map.count(t->op))
          << "Given tensor is not in the schedule plan";
      t = sch->stage_map[t->op]->op.output(t->value_index);
    }
    ret.push_back(t);
  }
  return ret;
}

// How to match pairs of (inputs, outputs) in the graph
std::vector<Operation> ExtractSubGraph(
    const Array<Operation>& roots,
    const ReadGraph& g,
    const Schedule& sch,
    std::unordered_map<const Node*, PlaceType>& dev,
    std::vector<Operation>& boundary,
    Array<Array<Tensor>>& inputs, 
    Array<Array<Tensor>>& outputs,
    std::vector<Operation>& merged_ops) {
   
  std::vector<Operation> workset;
  if (boundary.size() == 0) 
    return workset;

  // set up the search boundary 
  for (auto op : boundary) 
    workset.insert(workset.begin(), op);

  Array<Operation> input_ops;
  Array<Operation> output_ops;
  while (!workset.empty()) {
    Operation op = workset.back();
    workset.pop_back();
    Array<Tensor> input;
    Array<Tensor> output = {op.output(0)};
    output_ops.push_back(op);

    // remove nearest ancestors from workset
    auto anc = ExtractAncestors(op, g);
    for (Operation v : anc) { 
      auto it = std::find(workset.begin(), workset.end(), v);
      if (it != workset.end()) {
        workset.erase(it);
        input.push_back(v.output(0));
        input_ops.push_back(v);
      }
    }
    inputs.push_back(input);
    outputs.push_back(output);
    // LOG(INFO) << input << ":" << output;
    // GetSubGraph(RemapTensor(sch, output), 
    //             RemapTensor(sch, input), true);
  }

  std::vector<Operation> stack;
  std::vector<Operation> subgraph;
  std::unordered_set<const Node*> visited;
  for (Tensor t : outputs[0]) {
    stack.push_back(t->op);
    visited.insert(t->op.get());
  }

  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
    // auto node = static_cast<const OperationNode*>(v);
    // std::shared_ptr<OperationNode> opnode =
    //     std::make_shared<OperationNode>(node);
    // Operation op = Operation(opnode);
    subgraph.insert(subgraph.begin(), op);
    for (const auto& t : g.at(op)) { 
      bool reach_bound = false;
      for (auto& tensor : inputs[0]) {
        if (op.same_as(tensor->op))
          reach_bound = true;
      }
      
      if (t->op.defined()) {
        if (visited.count(t->op.get()) == 0) {
          visited.insert(t->op.get());
          if (!reach_bound) stack.push_back(t->op);
        }
      }
    }
  }

  // create aggregated op for subgraph (super stage)
  Stmt no_op = Evaluate::make(0);
  std::shared_ptr<ExternOpNode> aggregate =
        std::make_shared<ExternOpNode>();
  aggregate->name = "test"; 
  for (Operation op : input_ops) {
    if (auto extern_op = op.as<ExternOpNode>()) {
      for (auto& tensor : extern_op->inputs) {
        aggregate->inputs.push_back(tensor);
      }
      for (auto& buffer : extern_op->input_placeholders) {
        aggregate->input_placeholders.push_back(buffer);
      }
    }
  } 

  for (Operation op : output_ops) {
    if (auto extern_op = op.as<ExternOpNode>()) {
      for (auto& buffer : extern_op->output_placeholders) {
        aggregate->output_placeholders.push_back(buffer);
      }
    }
  }

  Stmt body = Evaluate::make(0);
  for (Operation op : subgraph) { 
    CHECK(op.as<ExternOpNode>());
    if (auto extern_op = op.as<ExternOpNode>()) {
      CHECK(extern_op->output_placeholders.size());
      Buffer out_buf = extern_op->output_placeholders[0];
      Stmt attr = AttrStmt::make(VarExpr(out_buf.node_), 
                      "attach_scope", StringImm::make("test"), no_op);
      body = Block::make(body, attr); 
    }
  }

  // decorate body with attr stmt
  Expr scope;
  CHECK(output_ops.size() > 0);
  CHECK(dev.count(output_ops[0].get()));
  switch (dev[output_ops[0].get()]) {
    case PlaceType::devHost : {
      scope = StringImm::make("cpu"); break;
    }
    case PlaceType::devFPGA : {
      scope = StringImm::make("fpga"); break;
    }
    case PlaceType::devGPU : {
      scope = StringImm::make("gpu"); break;
    }
  } 
  aggregate->body = AttrStmt::make(
      VarExpr(), "device_scope", scope, body);
  merged_ops.push_back(Operation(aggregate));
  return subgraph;
}

static int bound_index = 0;
// schedule the ops with subgraphs 
void PostDFSSplit(const Operation& op,
                  const ReadGraph& g,
                  std::unordered_set<Operation>* visited,
                  Array<Operation>* post_order,
                  std::unordered_map<const Node*, PlaceType>& dev,
                  std::vector<Operation>& subgraphs) {
  if (visited->count(op)) return;
  visited->insert(op);
  CHECK(dev.count(op.get())) << "not found " << op;

  // push op into separate list  
  bool reach_bound = false;
  for (auto& node : subgraphs) {
    if (op.same_as(node)) {
      // insert subgraph ops index 
      if (bound_index == 0)
        bound_index = visited->size();
      reach_bound = true;
    } 
  }

  for (const auto& t : g.at(op)) 
    PostDFSSplit(t->op, g, visited, post_order, dev, subgraphs);
  if (!reach_bound) post_order->push_back(op);
}

// propagate device info thru op trees 
Array<Operation> PostDFSSplit(
    const Array<Operation>& roots,
    const ReadGraph& g, const Schedule& sch) {

  // map from op to stage device scope 
  std::unordered_map<const Node*, PlaceType> dev;
  std::vector<Operation> boundary;
  std::unordered_set<Operation> visited;
  for (Operation op : roots) dev[op.get()] = devHost;
  for (Stage stage : sch->stages) {
    if (dev.count(stage->op.get()))
      CHECK(dev[stage->op.get()] == devHost)
        << "output " << stage << " should be placed on host scope";
    dev[stage->op.get()] = stage->device_type;
    if (stage->device_type != devHost) 
      boundary.insert(boundary.begin(), stage->op);
  }
  
  // propagate device inforation  
  // the inputs and outputs marked with xcel scope indicators
  // are required to form an enclosed subgraph  
  Array<Array<Tensor>> inputs, outputs;
  std::vector<Operation> merged_ops;
  auto subgraph = ExtractSubGraph(roots, g, sch, dev, 
                      boundary, inputs, outputs, merged_ops);

  Array<Operation> post_order;
  for (Operation op : roots) {
    PostDFSSplit(op, g, &visited, &post_order, dev, subgraph);
  }

  // op array index to insert subgraph 
  if (bound_index > 0) {
    Array<Operation> results;
    for (size_t k = 0; k < post_order.size(); k++) {
      if (k == post_order.size() - bound_index + 1) {
        for (auto& sub_op : subgraph)
          results.push_back(sub_op);
        for (auto& sub_op : merged_ops)
          results.push_back(sub_op);
      } 
      Operation op = post_order[k];
      results.push_back(op);
    }
    return results;
  }
  return post_order;
}

// Infer the palcement for each stage. The placement  
// 1. Subgraphs will be repalced with a Stage of KernelStmt
// 2. Group stages in xcel scope and create KernelDef
Schedule ScopePartition(const Schedule& sch) {

  Array<Operation> roots;
  for (Operation op : sch->outputs) 
    roots.push_back(sch->stage_map[op]->op);

  // map from tensor to ops 
  ReadGraph rmap;
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  // create read graph from roots
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
      // tensor as output of the operation 
      if (t->op.defined()) {
        Operation dep_op;
        CHECK(sch->stage_map.count(t->op))
          << "cannot find " << t->op;
        dep_op = sch->stage_map[t->op]->op;
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
  // CHECK(post_order.size() <= sch->stages.size());
  for (Operation op: post_order) {

    // FIXME: inconsistent stage address
    // CHECK(op2stage_.count(op.get()));
    // const Stage& s = op2stage_.at(op.get());

    Stage scopy;
    for (Stage s : sch->stages) {
      if (op->name == s->op->name) {
        if ((s->op.as<PlaceholderOpNode>() && op.as<PlaceholderOpNode>()) ||
            (s->op.as<ExternOpNode>() && op.as<ExternOpNode>())) {
          std::shared_ptr<StageNode> snode =
              std::make_shared<StageNode>(*s.operator->());
          scopy = Stage(snode);
          smap[s] = scopy;
          n->stages.push_back(scopy);
        }
      }
    }

    // merged stage hypernode
    if (!scopy.defined()) {
      Stage stage = Stage(op);
      n->stage_map.Set(op, stage);
      n->stages.push_back(stage);
    }
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

  // remaps op to new stage
  for (auto kv : sch->stage_map) { 
    if (smap.count(kv.second))
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
