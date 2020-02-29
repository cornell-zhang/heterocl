/*!
 *  Copyright (c) 2016 by Contributors
 * \file bound.cc
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
// update the attachment of stages with extern op
class AttachedStagesUpdater final : public IRVisitor {
  public:
    AttachedStagesUpdater(const Map<NodeRef, Stage>& stage_buf_map,
                          Stage& stage)
      : stage_buf_map_(stage_buf_map), stage_(stage) {};
  
    void Visit_(const AttrStmt* op) {
      if (op->attr_key == attr::attach_scope) {
        if (stage_buf_map_.count(op->node)) {
          // other stage's output used by current stage (and marked attached)
          stage_->attached_stages.push_back(stage_buf_map_.at(op->node));
        }
      }
      IRVisitor::Visit_(op);
    }
  
  private:
    const Map<NodeRef, Stage>& stage_buf_map_;
    Stage& stage_;
};

/*! \brief The graph context used during bound inference. */
struct GraphContext {
  /*! \brief The feed graph */
  FeedGraph feed_graph;
  /*! \brief Attachment path */
  AttachPath attach_path;
  /*! \brief The bind map */
  std::unordered_map<IterVar, IterVar> bind_map;
  /*! \brief map from op to stage */
  std::unordered_map<const Node*, Stage> op2stage_;
};

// schedule the ops 
void PostDFSSplit(const Operation& op,
                  const ReadGraph& g,
                  std::unordered_set<Operation>* visited,
                  Array<Operation>* post_order,
                  const Schedule& sch) {
  // skip broadcasting node 
  if (visited->count(op)) return;
  visited->insert(op);

  // insert parent op before self
  for (const auto& t : g.at(op)) {
    PostDFSSplit(t->op, g, visited, post_order, sch);
  }
  post_order->push_back(op);
}

// propagate device info thru op trees 
Array<Operation> PostDFSSplit(
    const Array<Operation>& roots,
    const ReadGraph& g, const Schedule& sch) {

  std::unordered_set<Operation> visited;
  Array<Operation> post_order;
  for (Operation op : roots) {
    PostDFSSplit(op, g, &visited, &post_order, sch);
  }
  return post_order;
}

// Infer the palcement for each stage. The placement  
// 1. Subgraphs will be repalced with a Stage of KernelStmt
// 2. Group stages in xcel scope and create KernelDef
Schedule ScopePartition(const Schedule& sch) {

  // feed graph maps from tensor to ops 
  Array<Operation> roots;
  std::unordered_map<Operation, Operation> opmap;
  for (Operation op : sch->outputs) { 
    roots.push_back(sch->stage_map[op]->op);
    opmap[sch->stage_map[op]->op] = op;
  }

  ReadGraph rmap;
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  // initialize the roots
  for (Operation op : roots) {
    if(!sch->stage_map.count(op) && sch->stage_map.size() > 0) {
      for (Operation out_op : sch->outputs) { 
        if (out_op->name == op->name) {
          opmap[op] = out_op;
          CHECK(sch->stage_map.count(out_op));
        }
      }
    }
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
        CHECK(sch->stage_map.count(t->op));
        dep_op = sch->stage_map[t->op]->op;
        if (dep_op.get() != t->op.get()) {
          opmap[dep_op] = t->op;
        }
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
   
  // resechdule ops array
  Array<Operation> post_order = 
     PostDFSSplit(roots, rmap, sch);
  std::unordered_set<Operation> output_set;
  for (Operation x : sch->outputs) {
    output_set.insert(x);
  }

  for (auto& op : post_order) {
    Stage stage;
    if (!sch->stage_map.count(op)) {
      CHECK(opmap.count(op)) << "invalid op " << op;
      stage = Stage(sch->stage_map[opmap[op]].node_);
    } else {
      stage = Stage(sch->stage_map[op].node_);
    }
  }

  // create new schedule 
  std::shared_ptr<ScheduleNode> n = std::make_shared<ScheduleNode>();
  std::unordered_map<Stage, Stage, NodeHash, NodeEqual> smap;
  n->outputs = sch->outputs;

  // copy the stages from array 
  for (Operation op: post_order) {
    if(!sch->stage_map.count(op)) {
      CHECK(opmap.count(op)) << "not found " << op;
      op = opmap[op];
      CHECK(sch->stage_map.count(op)) << "not found " << op;
    }
    Stage s = sch->stage_map[op]; 
    Stage scopy = Stage(s.node_);
    smap[s] = scopy;
    n->stages.push_back(scopy);
  }

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
    CHECK(smap.count(kv.second)) 
        << "not found " << kv.second;
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
