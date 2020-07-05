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
bool debug = false;

// TODO: construct sch->stage_buf_map_
// Update map from stage to its parent stages
class AttachingStagesUpdater final : public IRVisitor {
  public:
    AttachingStagesUpdater(
            std::unordered_map<std::string, std::string>& stage_parent_map)
      : stage_parent_map_(stage_parent_map) {};

    void Visit_(const AttrStmt* op) {
      if (op->attr_key == attr::attach_scope) {
          if (op->value.as<StringImm>() && op->node.as<BufferNode>()) {
            auto curr_stage_name = op->value.as<StringImm>()->value;
            auto child_stage_name = op->node.as<BufferNode>()->name;
            stage_parent_map_[child_stage_name] = curr_stage_name;
          }
      }
      IRVisitor::Visit_(op);
    }
    std::unordered_map<std::string, std::string>& stage_parent_map_;
};

// ir visitor to collect attached stages
class AttachedStagesFinder final : public IRVisitor {
  public:
    AttachedStagesFinder(std::unordered_set<std::string>& stage_list)
      : stage_list_(stage_list) {};
    void Visit_(const AttrStmt* op) {
      if (op->attr_key == attr::attach_scope) {
        if (auto buf = op->node.as<BufferNode>()) {
          stage_list_.insert(buf->name);
        }
      }
      IRVisitor::Visit_(op);
    }
    std::unordered_set<std::string>& stage_list_;
};

// extract child stages in extern module body
void TraceExternMods(const Array<Operation>& roots,
        const ReadGraph& g, 
        std::unordered_map<Operation, 
            std::unordered_set<std::string>>& extern_mods,
        /* children stage names defined in top stage body */ 
        std::unordered_set<std::string>& stage_list) {
  std::unordered_set<const Node*> visited;
  std::vector<Operation> stack;
  stack.push_back(roots[0]);
  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
    
    CHECK(g.count(op)) << "not found " << op;
    if (auto extern_op = op.as<ExternOpNode>()) {
      if (extern_op->body.as<ExternModule>()) {

        /* Record the input stages (child stages) of each
         * ExternModule op. These dependeing stages are saved 
         * into a map and will be checked layer when re-organing 
         * the attach_scope AttrStmt in the top stage body
         */
        for (const auto& t : g.at(op)) { 
          extern_mods[op].insert(t->op->name);
          if (g.count(t->op) && t->op->name.find(".new") == std::string::npos) {
            for (auto& pt : g.at(t->op)) {
              extern_mods[op].insert(pt->op->name);
            }
          }
        }
      }
      // capture the extern ops defined in top stage body
      // these ops should be excluded from subgraph
      if (op->name == "_top") {
        AttachedStagesFinder finder(stage_list);
        finder.Visit(extern_op->body);
      }
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

// create dfs post ordered attch attr stmt  
Stmt AttachScopeReorder(Array<Operation>& post_order,
        std::vector<Operation>& merged_ops,
        std::unordered_map<std::string, std::string>& stage_parent_map) {
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
      if (stage_parent_map.count(extern_op->name)) {
        if (stage_parent_map[extern_op->name] != "_top") {
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
      if (extern_op->name.find(".new") != std::string::npos) {
        CHECK_GT(i-1, 0) << "wrong op ordering fonud: " << post_order; 
        if (post_order[i-1]->name.find(".new") == std::string::npos) {
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

std::unordered_set<Operation> ExtractAncestors(Operation root, const ReadGraph& g) {
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  std::unordered_set<Operation> ops;
  stack.push_back(root);
  visited.insert(root.get());

  if (debug) {
    LOG(INFO) << "---------------------";
    for (auto& kv : g) {
      LOG(INFO) << "------------";
      LOG(INFO) << kv.first;
      for (auto& t : kv.second) LOG(INFO) << t->op;
    }
  }

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
// create an aggregate super stage (in merged op)
std::vector<Operation> ExtractSubGraph(
    const Array<Operation>& roots,
    const ReadGraph& g,
    const Schedule& sch,
    std::unordered_map<const Node*, DeviceType>& dev,
    // module map recording super stage attachment 
    std::unordered_map<Operation, 
        std::unordered_set<std::string>> atts_map,
    std::vector<Operation>& boundary,
    Array<Array<Tensor>>& inputs, 
    Array<Array<Tensor>>& outputs,
    std::vector<Operation>& merged_ops,
    std::unordered_set<std::string> stage_list, 
    std::unordered_map<std::string, std::string>& stage_parent_map) {
   
  std::vector<Operation> workset;
  if (boundary.size() == 0) return workset;

  // set up the search boundary 
  for (auto op : boundary) {
    workset.insert(workset.begin(), op);
  }

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
    CHECK(input.size() > 0) 
      << "cannot found boundary for output " << output;
    // GetSubGraph(RemapTensor(sch, output), 
    //             RemapTensor(sch, input), true);
  }

  // traverse the whole graph
  std::vector<Operation> stack;
  std::vector<Operation> subgraph;
  std::unordered_set<const Node*> visited;
  for (Tensor t : outputs[0]) {
    stack.push_back(t->op);
    visited.insert(t->op.get());
  }

  CHECK(!stack.empty());
  std::unordered_set<const Node*> shared;
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
        if (op.same_as(tensor->op)) {
          reach_bound = true;
        }
      }
      
      if (t->op.defined()) {
        if (visited.count(t->op.get()) == 0) {
          visited.insert(t->op.get());

          if (!reach_bound) {
            // skip the op in the subgraph if it was declared 
            // on the host as ane extern op, and used in the subgraph
            if (g.at(t->op).size() == 0 && 
                stage_list.find(t->op->name) != stage_list.end()) {
              // continue;
            }
            stack.push_back(t->op);
          }
        } else { // visited ancestor
          shared.insert(t->op.get());
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

  // create empty buffer node for aggregate super stage
  Buffer aggregate_buffer = BufferNode::make(Var(aggregate->name, Handle()),
      Int(32), Array<Expr>(), Array<Expr>(), Expr(), aggregate->name, "", 0, 0);
  aggregate->output_placeholders.push_back(aggregate_buffer);

  // Fix: create a map recording original stage order
  // Since some dependencies are not captured by HeteroCL, the map
  // is used to record the original order of stages in the schedule
  std::vector<std::string> order_subgraph_op;
  std::unordered_set<std::string> subgraph_op_names;
  for (auto& op : subgraph) {
    std::string name = op->name;
    CHECK(!subgraph_op_names.count(name)) << name;
    subgraph_op_names.insert(name);
  } 
  for (Stage stage : sch->stages) {
    if (subgraph_op_names.count(stage->op->name)) { 
      order_subgraph_op.push_back(stage->op->name);
    }
  }
  std::vector<Operation> reordered_subgraph;
  CHECK(subgraph_op_names.size() == subgraph.size());
  CHECK(order_subgraph_op.size() == subgraph.size())
      << "Please checking the variable naming, "
      << "More than one tensors sharing the same name";
  // reorder the ops  
  for (auto& name : order_subgraph_op) {
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

  /** Re-arrange the op in subgraph
   *  1. Append .new ops in the front 
   *  2. Consider the partitioned op attachement 
   *  3. Reorder the ops in te subgraph based on sch->stages
  */
  CHECK(reordered_subgraph.size() > 0);
  std::vector<Operation> new_subgraph;
  size_t op_count = 0;
  for (auto& op : reordered_subgraph) {
    auto name = op->name;
    if (name.find(".new") != std::string::npos) {
      new_subgraph.insert(new_subgraph.begin(), op);
      op_count += 1;

      // check attached partition node
      // e.g. A.channel -> A.new / A.new.partition 
      for (auto& op_tensor_kv : g) {
        if (op_tensor_kv.first->name == name + ".partitioned") {
          op_count += 1;
          new_subgraph.insert(new_subgraph.begin(), op_tensor_kv.first);
          if (debug) LOG(INFO) << "buffer " << name << " partitioned on device";
        }
      }

    // ordinary operators
    } else { 
      // insert shared ops in the front (e.g. scalars...)
      if (shared.find(op.get()) != shared.end()) {
        new_subgraph.insert(new_subgraph.begin() + op_count, op);
        op_count += 1;
        continue;
      }
      new_subgraph.push_back(op);
    }
  }

  // find the updated tensors in extern mod subgraph 
  // insert the extern mod into aggregate node
  std::unordered_map<Operation, int> inserted;
  std::unordered_map<Operation, std::unordered_set<std::string>> op2modifed;

  // record the modified ops in subgraph
  std::unordered_set<std::string> nodes;
  for(auto op : new_subgraph) {
    nodes.insert(op->name);
  }
  for(auto& kv : atts_map) {
    inserted[kv.first] = 0;
    // LOG(INFO) << kv.first << ":------------";
    // for (auto& k : kv.second) LOG(INFO) << k;
    for (auto& v : kv.second) {
      if (nodes.find(v) != nodes.end()) {
        if (v.find(".new") == std::string::npos) {
          op2modifed[kv.first].insert(v);
        }
      }
    }
  }

  Stmt body = Evaluate::make(0);
  for (Operation op : new_subgraph) { 
    CHECK(op.as<ExternOpNode>()) << op;
    if (auto extern_op = op.as<ExternOpNode>()) {

      if (extern_op->name.find(".partitioned") != std::string::npos)
        continue;

      // check if subgraph op in extern module inputs
      // the extern module acts as upadter of these ops 
      bool updated_op = false;
      for (auto& kv : op2modifed) {
        if (kv.second.count(op->name) && 
            op->name.find(".new") == std::string::npos) {

          updated_op = true;
          inserted[kv.first] += 1;

          // insert extern module op after its dependent stages
          if (inserted[kv.first] == (signed)kv.second.size()) {
            auto mod_op = kv.first.as<ExternOpNode>();
            Buffer mod_buf = mod_op->output_placeholders[0];
            Stmt attr = AttrStmt::make(VarExpr(mod_buf.node_), 
                            "attach_scope", StringImm::make("test"), no_op);
            body = Block::make(body, attr); 
          }
        }
      }

      // insert standalone subgraph op 
      if (updated_op) continue;
      // continue if the op already has an attaching scope
      if (stage_parent_map.count(extern_op->name)) {
        if (stage_parent_map[extern_op->name] != "_top")
          continue;
      }

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
    case DeviceType::devHost : {
      scope = StringImm::make("cpu"); break;
    }
    case DeviceType::devFPGA : {
      scope = StringImm::make("fpga"); break;
    }
    case DeviceType::devGPU : {
      scope = StringImm::make("gpu"); break;
    }
  } 
  aggregate->body = AttrStmt::make(
      VarExpr(), attr::device_scope, scope, body);

  if (debug) {
    for(auto op : new_subgraph) LOG(INFO) << op;
    LOG(INFO) << aggregate->body;
  }

  merged_ops.push_back(Operation(aggregate));
  return new_subgraph;
}

// extract the bounded op arrays from subgraph root
// needed to add to extracted subgrapg ( since subgraph 
// does not capture the ops in extern module )
void PostDFSBoundary(const Operation& op,
        const ReadGraph& g,
        std::unordered_set<Operation>* visited,
        Array<Operation>* post_order,
        Array<Operation>* bounded_ops,
        std::unordered_map<Operation, 
            std::unordered_set<std::string>>& extern_mods,
        std::unordered_set<std::string>& sub_ops,
        std::unordered_set<std::string>& stage_list) {
  if (visited->count(op)) return;
  visited->insert(op);

  // CHECK(op.as<ExternOpNode>()) << op;
  for (const auto& t : g.at(op)) {
    PostDFSBoundary(t->op, g, visited, post_order, 
        bounded_ops, extern_mods, sub_ops, stage_list);
  }

  if (op.as<PlaceholderOpNode>()) {
    post_order->push_back(op);
    return;
  }

  // record ops before .new ops
  bool in_ext_mod = false;
  if (op.as<ExternOpNode>()->body.as<ExternModule>()) in_ext_mod = true;
  for (auto& kv : extern_mods) {
    // the op required to be a child stage
    if ((kv.second.find(op->name) != kv.second.end()) &&
        // ignore the moved tensor (part of test stage)
        (op->name.find(".new") == std::string::npos)  &&
        // should be part of subgraph
        (sub_ops.find(op->name) == sub_ops.end()))
      in_ext_mod = true;
  }

  if (in_ext_mod)
    bounded_ops->push_back(op);

  // record ops outside subgraph
  if ((sub_ops.find(op->name) == sub_ops.end()) &&
      (!op.as<ExternOpNode>()->body.as<ExternModule>())) {
    for (auto& kv : extern_mods) {
      // the op should not be a child stage of extern modules
      if (kv.second.find(op->name) == kv.second.end()) {
        post_order->push_back(op);
      }
      // add the op to post_order, if it appears in top stage body
      if (stage_list.find(op->name) != stage_list.end()) {
        // post_order->push_back(op);
        // LOG(INFO) << op->name;
      }
    }
  }
}

// schedule the ops with subgraphs 
// store ops that are not in subgraph
void PostDFSSplit(const Operation& op,
                  const ReadGraph& g,
                  std::unordered_set<Operation>* visited,
                  Array<Operation>* post_order,
                  std::unordered_map<const Node*, DeviceType>& dev,
                  std::vector<Operation>& subgraphs) {
  if (visited->count(op)) return;
  visited->insert(op);
  CHECK(dev.count(op.get())) << "not found " << op;

  // visit from root to source and record break point  
  // push op into array if it is outside the subgraph
  bool reach_bound = false;
  for (auto& node : subgraphs) {
    if (op.same_as(node)) {
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
  std::unordered_map<const Node*, DeviceType> dev;
  std::vector<Operation> boundary;
  std::unordered_set<Operation> visited;

  std::unordered_map<Operation, 
      std::unordered_set<std::string>> extern_mods;
  for (Operation op : roots) { 
    dev[op.get()] = DeviceType::devHost;
  }

  // check the external module
  std::unordered_set<std::string> stage_list;
  TraceExternMods(roots, g, extern_mods, stage_list);
  // collect dev info and attachment info
  std::unordered_map<std::string, std::string> stage_parent_map;
  AttachingStagesUpdater updater(stage_parent_map);

  for (Stage stage : sch->stages) {
    if (auto extern_op = stage->op.as<ExternOpNode>()) {
      updater.Visit(extern_op->body);
    }
    if (dev.count(stage->op.get()))
      CHECK(dev[stage->op.get()] == DeviceType::devHost)
        << "output " << stage << " should be placed on host scope";
    dev[stage->op.get()] = stage->device_type;
    if (stage->device_type != DeviceType::devHost) {
      boundary.insert(boundary.begin(), stage->op);
    }
  }
  
  // propagate device inforation  
  // the inputs and outputs marked with xcel scope indicators
  // are required to form an enclosed subgraph  
  Array<Array<Tensor>> inputs, outputs;
  std::vector<Operation> merged_ops;

  // not create aggregate node for extern module 
  // note: the subgraph does not exactly descibe the compute flow
  // e.g. if there are some other super stages modifying the tensor 
  // before we use the tensor, the read graph does not capture that
  auto subgraph = ExtractSubGraph(roots, g, sch, dev, extern_mods, 
                      boundary, inputs, outputs, merged_ops, stage_list, stage_parent_map);

  // for (auto& op : subgraph) LOG(INFO) << op;
  Array<Operation> post_order;
  Array<Operation> bounded_ops; 
  for (Operation op : roots) {
    if (extern_mods.size() > 0) {
      // create op array of extern module (from .new to super stage root) 
      // i.e. inner ops inside extern module (must be bounded by .new ops) 
      // the result is returned in bounded_ops

      bool dev_scope = false;
      // TODO: consider multiple extern modules
      for (auto& kv : extern_mods) {
        for (auto& input : kv.second) {
          if (input.find(".new") != std::string::npos)
            dev_scope = true;
        }
      }

      if (dev_scope) {
        LOG(INFO) << "input tensors in device scope";
        std::unordered_set<Operation> visited_ops;
        // extract all stages in extern module for updating logic
        std::unordered_set<std::string> sub_ops;

        for (auto& op : subgraph) {
          sub_ops.insert(op->name);
        }
        // extract bounded_ops (ops of extern module that are within the subgraph)
        PostDFSBoundary(op, g, &visited_ops, &post_order, 
            &bounded_ops, extern_mods, sub_ops, stage_list);

      } else { 
        LOG(WARNING) << "input tensors of IP core on host scope (sim mode only)";
        PostDFSSplit(op, g, &visited, &post_order, dev, subgraph);
      }
    } else { 
      // without extern module (subgraph & post_order)
      // return post_order with op out of subgraph
      PostDFSSplit(op, g, &visited, &post_order, dev, subgraph);
    }
  }

  // op array index to insert subgraph 
  // for (auto& op : subgraph) LOG(INFO) << op;
  bool inserted = false;
  if (merged_ops.size() > 0) {
    Array<Operation> results;
    for (size_t k = 0; k < post_order.size(); k++) {
      // fix: insert right before the first .new
      // if (k == post_order.size() - (bound_index - 1))
      auto sname = post_order[k]->name;
      if (!inserted && sname.find(".new") != std::string::npos) {
        inserted = true;
        // LOG(INFO) << "insert beofre " << post_order[k];

        if (extern_mods.size() == 0) {
          for (auto& sub_op : subgraph) {
            results.push_back(sub_op);
          }
          for (auto& sub_op : merged_ops) { 
            results.push_back(sub_op);
          }

        // replace the modfied tensor ops with extern module
        // i.e. ops in the keys of corresponding module
        } else { 

          // subgraph with incomplete updating rules
          for (auto& sub_op : subgraph) {
            results.push_back(sub_op);
          }

          // all the ops in ExternModules bounded by new.op
          // excluding the ops included in subgraph
          CHECK(bounded_ops.size() > 0);
          for (Operation op : bounded_ops) {
            results.push_back(op);
          }

          // aggreated op with empty body
          for (auto& sub_op : merged_ops) { 
            results.push_back(sub_op);
          }
        }
      } 
      Operation op = post_order[k];

      // fix: re-arrange attr stmt inside 
      if (op->name == "_top") {
        Stmt no_op = Evaluate::make(0);
        std::shared_ptr<ExternOpNode> new_op =
              std::make_shared<ExternOpNode>();
        new_op->name = op->name; 
        // top op input / output buffers
        auto extern_op = op.as<ExternOpNode>();
        CHECK(extern_op) << "invalid _top op node";
        new_op->inputs = std::move(extern_op->inputs);
        new_op->input_placeholders = std::move(extern_op->input_placeholders);
        new_op->output_placeholders = std::move(extern_op->output_placeholders);
        // rearrange attachment scope attr inside _top body
        new_op->body = AttachScopeReorder(post_order, merged_ops, stage_parent_map);
        op = Operation(new_op);
      }
      results.push_back(op);
    }
    CHECK(results.size() >= sch->stages.size())
      << "missing ops in result. size " << results.size() << ":" << sch->stages.size()
      << results;
    return results;
  }

  return post_order;
}

// Infer the palcement for each stage. The placement  
// 1. Subgraphs will be repalced with a Stage of KernelStmt
// 2. Group stages in xcel scope and create KernelDef
Schedule ScopePartition(const Schedule& sch) {

  Array<Operation> roots;
  for (Operation op : sch->outputs) {
    if (sch->stage_map[op]->is_output)
      roots.push_back(sch->stage_map[op]->op);
  }
  CHECK(!roots.empty()) << "empty roots";

  // map from tensor to ops 
  ReadGraph rmap;
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  // create read graph from roots
  for (Operation op : roots) {
    stack.push_back(op);
    visited.insert(op.get());
  }

  // for (auto& kv : sch->stage_map)
  //   LOG(INFO) << kv.first << ":" << kv.second;
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

  // re-sechdule ops array if subgraph exists
  Array<Operation> post_order = PostDFSSplit(roots, rmap, sch);
  if (post_order.size() == sch->stages.size()) return sch;

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
