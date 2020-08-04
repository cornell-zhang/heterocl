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

bool debug = true;

using namespace ir;

using std::vector;
using std::string;
using std::unordered_set;
using std::unordered_map;

// Store the stage placement information 
struct DevScope {
  DeviceType dev_type;
  bool is_endpoint{false};
};

// Update map from stage to its parent stage
class AttachingStagesUpdater final : public IRVisitor {
  public:
    AttachingStagesUpdater(
            unordered_map<string, string>& stage_to_attach_parent,
            unordered_map<string, vector<string>>& stage_to_attach_children)
      : stage_to_attach_parent_(stage_to_attach_parent) {};

    // Schedule op function check the child stage buffer
    // and attach the matched stage to its position 
    void Visit_(const AttrStmt* op) {
      if (op->attr_key == attr::attach_scope) {
          if (op->value.as<StringImm>() && op->node.as<BufferNode>()) {
            auto curr_stage_name = op->value.as<StringImm>()->value;
            auto child_stage_name = op->node.as<BufferNode>()->name;
            stage_to_attach_parent_[child_stage_name] = curr_stage_name;
            LOG(INFO) << "Stage " << child_stage_name << " attaching to "
                << curr_stage_name << "...";
          }
      }
      IRVisitor::Visit_(op);
    }
    unordered_map<string, string>& stage_to_attach_parent_;
};


// create dfs post ordered attch attr stmt  
Stmt AttachScopeReorder(vector<Operation>& post_order,
        vector<Operation>& merged_ops,
        unordered_map<string, string>& stage_to_attach_parent) {
  Stmt body;
  Stmt no_op = Evaluate::make(0);
  CHECK(post_order.size() > 0);

  for (int i = post_order.size() - 1; i >= 0; i--) {
    auto& op = post_order[i];
    if (auto extern_op = op.as<ExternOpNode>()) {
      Buffer buf = extern_op->output_placeholders[0];
      if (extern_op->name == "_top") {
        continue;
      }

      // stage that has an original attach point
      if (stage_to_attach_parent.count(extern_op->name)) {
        if (stage_to_attach_parent[extern_op->name] != "_top") {
          continue;
        }
      }
      if (!body.defined()) {
        body = AttrStmt::make(VarExpr(buf.node_), 
                attr::attach_scope, StringImm::make("_top"), no_op);
      } else {
        body = AttrStmt::make(VarExpr(buf.node_), 
                attr::attach_scope, StringImm::make("_top"), body);
      }
      // find the right place to insert the arrgragated super-stage 
      // op into the top-level stage body. It should be inserted right before
      // the last .new tensors (which indiactes the end od xcel scope)
      if (extern_op->name.find(".new") != string::npos) {
        // CHECK_GT(i-1, 0) << "wrong op ordering fonud: " << post_order; 
        if (post_order[i-1]->name.find(".new") == string::npos) {
          for (auto& sub_op : merged_ops) { 
            auto sub_ext_op = sub_op.as<ExternOpNode>();
            Buffer sub_buf = sub_ext_op->output_placeholders[0];
            CHECK(body.defined());
            body = AttrStmt::make(VarExpr(sub_buf.node_), 
                attr::attach_scope, StringImm::make("_top"), body);
            if (debug) LOG(INFO) << "\nreordered top stage body:\n" << body;
          }
        }
      }
    }
  }
  CHECK(body.defined());
  return body;
}

// Extract all ancestor ops of a specific root
unordered_set<Operation> ExtractAncestors(Operation root, const ReadGraph& g) {

  vector<Operation> stack;
  unordered_set<const Node*> visited;
  unordered_set<Operation> ops;
  stack.push_back(root);
  visited.insert(root.get());

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


// How to match pairs of (inputs, outputs) in the graph
// create an aggregate super stage (in merged op)
vector<Operation> ExtractSubGraph(
    const Array<Operation>& roots,
    const ReadGraph& g,
    const Schedule& sch,
    unordered_map<const Node*, DevScope>& dev,
    vector<Operation>& boundary,
    vector<Operation>& merged_ops,
    vector<Operation>& non_subgraph_ops,
    unordered_map<string, string>& stage_to_attach_parent) {
   
  // Debug: print the read graph
  if (debug) {
    LOG(INFO) << "------------ Read Graph -------------";
    for (auto& kv : g) {
      LOG(INFO) << "------------";
      LOG(INFO) << "Stage " << kv.first << " reads from ";
      for (auto& t : kv.second) 
        LOG(INFO) << "    " << t->op;
    }
  }

  Array<Array<Tensor>> inputs, outputs;
  vector<Operation> workset;
  if (boundary.size() == 0) return workset;

  // Set up the search boundary 
  for (auto op : boundary) {
    LOG(INFO) << "Insert boundary op " << op->name << "...";
    workset.insert(workset.begin(), op);
  }

  Array<Operation> input_ops;
  Array<Operation> output_ops;

  // Assume that the subgraph only has 
  // a single output root, and the subgraph's 
  // boundary is well defined by applying .to() 
  while (!workset.empty()) {
    Operation op = workset.back();
    workset.pop_back();
    Array<Tensor> input;
    Array<Tensor> output = {op.output(0)};
    output_ops.push_back(op);

    auto ancestors = ExtractAncestors(op, g);
    for (Operation v : ancestors) { 
      auto it = std::find(workset.begin(), workset.end(), v);
      if (it != workset.end()) {
        LOG(INFO) << "Identify input tensor " << v->name 
            << " of root tensor " << op->name;
        workset.erase(it);
        input.push_back(v.output(0));
        input_ops.push_back(v);
      }
    }
    inputs.push_back(input);
    outputs.push_back(output);
    CHECK(input.size() > 0) 
      << "Cannot found boundary for output " << output 
      << ". Make sure the input tensors are moved to FPGA correctly...";
  }

  // Traverse the graph to find the ops that 
  // are within the boundary. Save these ops into 
  // the op array (i.e. subgraph) 
  vector<Operation> stack;
  vector<Operation> subgraph;
  unordered_set<const Node*> visited;
  for (Tensor t : outputs[0]) {
    stack.push_back(t->op);
    visited.insert(t->op.get());
  }

  CHECK(!stack.empty());
  unordered_set<const Node*> shared;
  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
 
    // Save op into the subgraph
    LOG(INFO) << "Add op " << op << " to the subgraph..."; 
    subgraph.insert(subgraph.begin(), op);
    for (const auto& t : g.at(op)) { 

      // Check whether op and its predecessors 
      // are in the graph or not. If the 
      bool reach_bound = false;
      CHECK(dev.count(t->op.get()));
      if (dev[t->op.get()].is_endpoint) {
          LOG(INFO) << "    Endpoint found " << t << "... Setup reach_bonud";
          reach_bound = true;
      }

      if (t->op.defined()) {
        if (visited.count(t->op.get()) == 0) {
          visited.insert(t->op.get());

          if (!reach_bound) {
            stack.push_back(t->op);
          }
        } else { // visited ancestor
          shared.insert(t->op.get());
        }
      }
    }
  }

  // Create aggregated op for subgraph (super stage)
  Stmt no_op = Evaluate::make(0);
  std::shared_ptr<ExternOpNode> aggregate = std::make_shared<ExternOpNode>();
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

  // Create empty buffer node for aggregate super stage
  Buffer aggregate_buffer = BufferNode::make(Var(aggregate->name, Handle()),
      Int(32), Array<Expr>(), Array<Expr>(), Expr(), aggregate->name, "", 0, 0);
  aggregate->output_placeholders.push_back(aggregate_buffer);

  // Create an attach point for all the in-subgraph ops
  // To preserve the order that is implicitly enforced 
  vector<string> order_subgraph_ops;
  unordered_set<string> subgraph_op_names;
  for (auto& op : subgraph) {
    string name = op->name;
    CHECK(!subgraph_op_names.count(name)) << name;
    subgraph_op_names.insert(name);
  } 
  for (Stage stage : sch->stages) {
    if (subgraph_op_names.count(stage->op->name)) { 
      order_subgraph_ops.push_back(stage->op->name);
    } else {
      non_subgraph_ops.push_back(stage->op);
    }
  }
  vector<Operation> reordered_subgraph;
  CHECK(subgraph_op_names.size() == subgraph.size());
  CHECK(order_subgraph_ops.size() == subgraph.size())
      << "Please checking the variable naming: "
      << "More than one tensors sharing the same name";
  for (auto& name : order_subgraph_ops) {
    bool found_op = false;
    for (auto& op : subgraph) {
      if (op->name == name) {
        found_op = true;
        reordered_subgraph.push_back(op);
        break;
      }
    }
    CHECK(found_op);
  } 
  CHECK(reordered_subgraph.size() == subgraph.size());

  // Create the aggregate node body 
  // First check whether the stage has a parent stage other than _top
  // Avoid creating multiple attaching points for the same stage 
  Stmt body = Evaluate::make(0);
  for (int i = reordered_subgraph.size() - 1; i >= 0; i--) { 
    auto op = reordered_subgraph[i];
    CHECK(op.as<ExternOpNode>()) << op;
    auto extern_op = op.as<ExternOpNode>();

    // Continue if the op already has an attaching scope
    if (stage_to_attach_parent.count(extern_op->name)) {
      if (stage_to_attach_parent[extern_op->name] != "_top") {
        LOG(INFO) << "INFO : the stage " << extern_op->name
            << " has a (non-top) parent stage " << stage_to_attach_parent[extern_op->name];
        continue;
      }
    }

    CHECK(extern_op->output_placeholders.size() > 0);
    Buffer out_buf = extern_op->output_placeholders[0];
    body = AttrStmt::make(VarExpr(out_buf.node_), 
                    "attach_scope", StringImm::make("test"), body);
  }

  // decorate body with attr stmt
  Expr scope = StringImm::make("fpga"); 
  CHECK(output_ops.size() > 0);
  CHECK(dev.count(output_ops[0].get()));
  aggregate->body = AttrStmt::make(VarExpr(), attr::device_scope, scope, body);

  if (debug) {
    LOG(INFO) << "----------- Ops in Subgraph --------------";
    for(auto op : reordered_subgraph) 
        LOG(INFO) << "    " << op;
    LOG(INFO) << aggregate->body;
  }

  merged_ops.push_back(Operation(aggregate));
  return reordered_subgraph;
}


// Analyze the existing sch->ops and readgraph
// Extracting the graph hierarchy
Array<Operation> HostDevPartition(
    const Array<Operation>& roots,
    const ReadGraph& g, const Schedule& sch) {

  // Map from op(node) to its device type
  unordered_map<const Node*, DevScope> dev;
  vector<Operation> boundary;
  unordered_set<Operation> visited;

  // The roots ops must be placed on host
  for (Operation op : roots) { 
    DevScope scope;
    scope.dev_type = DeviceType::devHost;
    dev[op.get()] = scope;
  }

  // Map from a stage to its parent stage's name
  // There shuold not any duplicates of stage names
  unordered_map<string, string> stage_to_attach_parent;
  unordered_map<string, vector<string>> stage_to_attach_children;
  AttachingStagesUpdater updater(stage_to_attach_parent, stage_to_attach_children);

  for (Stage stage : sch->stages) {
    // LOG(INFO) << "Checking stage " << stage << " (" 
    //   << static_cast<int>(stage->device_type) << ")";
    if (auto extern_op = stage->op.as<ExternOpNode>()) {
      updater.Visit(extern_op->body);
    }

    if (dev.count(stage->op.get())) {
      CHECK(dev[stage->op.get()].dev_type == DeviceType::devHost)
        << "output " << stage << " should be placed on host scope";
    }

    // Create an array of op sitting on the bonudary 
    // The memory interface information is saved into 
    // the "endpoint" field of the target stage
    DevScope info;
    if (stage->endpoint.defined()) {
        LOG(INFO) << "Endpoint stage " << stage;
        info.is_endpoint = true;
    }
    info.dev_type = stage->device_type;
    dev[stage->op.get()] = info;
    if (stage->device_type != DeviceType::devHost) {
      boundary.insert(boundary.begin(), stage->op);
    }
  }
  
  // Save the super-stage op in merged_ops 
  vector<Operation> merged_ops;
  vector<Operation> non_subgraph_ops;

  // Note: the subgraph does not exactly descibe the compute flow
  // e.g. if there are some other super stages modifying the tensor 
  // before we use the tensor, the read graph does not capture that
  auto subgraph = ExtractSubGraph(roots, g, sch, dev, 
                      boundary, merged_ops, non_subgraph_ops, 
                      stage_to_attach_parent);

  // Insert the subgraph ops and the super stage op. Also update
  // the top stage body in different cases
  //   1. The subgraph ops are not part of any super stages
  //   2. The subgraph contains some super stages
  //   3. The subgraph has intersection with other super stages (Not supported) 
  if (merged_ops.size() > 0) {
    vector<Operation> results;

    // First insert all the endpoint stages
    // Insert the new aggregate op after last endpoint stage
    int insert_point = 0;
    for (size_t m = 0; m < non_subgraph_ops.size(); m++) {
        auto op = non_subgraph_ops[m];
        if (dev[op.get()].is_endpoint) {
            insert_point = m + 1;
            LOG(INFO) << "Ops (ep) outside the subgraph : " << non_subgraph_ops[m];
        } else {
            LOG(INFO) << "Ops outside the subgraph : " << non_subgraph_ops[m];
        }
        results.push_back(op);
    }

    LOG(INFO) << "INFO: Final insert point index : " << insert_point;
    for (size_t k = 0; k < subgraph.size(); k++) {
      Operation op = subgraph[k];
      LOG(INFO) << "Ops in the subgraph : " << op;
      results.insert(results.begin() + insert_point, op);

      // Rearrange attachment scope attr inside _top body
      // We should change the attach attr stmts 
      // if (op->name == "_top") {
      //   Stmt no_op = Evaluate::make(0);
      //   std::shared_ptr<ExternOpNode> new_op = std::make_shared<ExternOpNode>();
      //   new_op->name = op->name; 

      //   auto extern_op = op.as<ExternOpNode>();
      //   CHECK(extern_op) << "The top op node is not an ExternOpNode...";
      //   new_op->inputs = std::move(extern_op->inputs);
      //   new_op->input_placeholders = std::move(extern_op->input_placeholders);
      //   new_op->output_placeholders = std::move(extern_op->output_placeholders);

      //   new_op->body = AttachScopeReorder(post_order, merged_ops, stage_to_attach_parent);
      //   op = Operation(new_op);
      // }
      // results.push_back(op);
    }

    // Insert the merged op
    for (auto& op : merged_ops) {
      int index = insert_point + subgraph.size();
      results.insert(results.begin() + index, op);
    }

    Array<Operation> post_order;
    for (auto& op : results) post_order.push_back(op);

    CHECK(results.size() == sch->stages.size() + merged_ops.size())
      << "Missing ops in result. size " << results.size() << ":" << sch->stages.size()
      << post_order;

    LOG(INFO) << post_order;
    LOG(INFO) << "Original op array " << PostDFSOrder(roots, g);
    return post_order;
  } 

  // There is no enclosure found in the ops
  // Retuns the origin DFS ordered op array
  return PostDFSOrder(roots, g);
}


// This is the entry point of stage reordering
// The scope partition function will 
//   1. Create a super stage to attach all the stages that 
//      are placed on FPGA. 
//   2. Create a read graph to trace the stage hierarchy. We traverse
//      through the top level stage body to build a ordered map.  
Schedule ScopePartition(const Schedule& sch) {

  // Extract the root ops of CDFG
  Array<Operation> roots;
  for (Operation op : sch->outputs) {
    if (sch->stage_map[op]->is_output)
      roots.push_back(sch->stage_map[op]->op);
  }
  CHECK(!roots.empty()) << "empty roots";

  ReadGraph rmap;
  vector<Operation> stack;
  unordered_set<const Node*> visited;

  // Create read graph from roots
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

        // Skip the tensor if it has been visited 
        // E.g. a tensor multicast to multiple consumers
        if (visited.count(dep_op.get()) == 0) {
          visited.insert(dep_op.get());
          stack.push_back(dep_op);
        }
      }
    }
    rmap.Set(op, new_deps);
  }

  // Create a map from op to stage that can
  // be used to backtrace stage from ops
  unordered_map<const Node*, Stage> op2stage_;
  for (Stage stage : sch->stages) {
    std::shared_ptr<StageNode> snode = std::make_shared<StageNode>(*stage.operator->());
    op2stage_[stage->op.get()] = Stage(snode);
  }

  // Re-sechdule ops array if subgraph exists.
  // The HostDevPartition() function creates new super stage(op) 
  // that contains the attaching points for its children stages 
  Array<Operation> post_order = HostDevPartition(roots, rmap, sch);
  if (post_order.size() == sch->stages.size()) return sch;

  unordered_set<Operation> output_set;
  for (Operation x : sch->outputs) {
    output_set.insert(x);
  }

  // create new schedule node 
  std::shared_ptr<ScheduleNode> n = std::make_shared<ScheduleNode>();
  unordered_map<Stage, Stage, NodeHash, NodeEqual> smap;
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
          // replace stage op body for _top
          if (scopy->op->name == "_top") {
            // LOG(INFO) << scopy;
            // LOG(INFO) << op.as<ExternOpNode>()->body;
            // LOG(INFO) << scopy->op.as<ExternOpNode>()->body;
            scopy = Stage(op);
            n->stage_map.Set(op, scopy);
          }
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
