/*!
 *  Copyright (c) 2020 by Contributors
 * \file schedule_reorder.cc
 * \brief The bound inference logic.
 */
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>
#include <unordered_map>
#include <unordered_set>
#include "../runtime/thread_storage_scope.h"
#include "./graph.h"
#include "./message_passing.h"

namespace TVM {
namespace schedule {

bool debug = true;

using namespace ir;

using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;

// Store the stage placement information
struct DevScope {
  DeviceType dev_type;
  bool is_endpoint{false};
  StorageType storage_type;
  StreamType stream_type;
  int mem_port{-1};
  int channel_depth{-1};
  int burst_len{-1};
  string target_tensor;
};

// (Top level) Basic Blocks in stage body. The types can be
// 1. AtttStmt attachments
// 2. Nested for loops
struct BlockElem {
  string name;
  bool is_nested_loops{false};
  bool is_attach_point{false};
  int attach_point_loop_level{-1};
};

// Update map from stage to its parent stage
class AttachingStagesUpdater final : public IRVisitor {
 public:
  AttachingStagesUpdater(
      unordered_map<string, string>& stage_to_attach_parent,
      unordered_map<string, vector<BlockElem>>& stage_to_attach_children)
      : stage_to_attach_parent_(stage_to_attach_parent),
        stage_to_attach_children_(stage_to_attach_children) {}

  // Schedule op function check the child stage buffer
  // and attach the matched stage to its position
  void Visit_(const AttrStmt* op) {
    if (op->attr_key == attr::attach_scope) {
      if (op->value.as<StringImm>() && op->node.as<BufferNode>()) {
        auto curr_stage_name = op->value.as<StringImm>()->value;
        auto child_stage_name = op->node.as<BufferNode>()->name;
        stage_to_attach_parent_[child_stage_name] = curr_stage_name;

        // The attaching points can be at the top level of stage body
        // e.g. partition or inside the loop body (e.g. reuse)
        if (curr_stage_name != input_stage_name) {
          // the newly created (PE) stage contains an attaching sub-stage
          // which does not belong to it originally. In such a case we just
          // ignore the the attaching and analysis
          LOG(WARNING)
              << " Found a duplicate child stage attaching to multiple "
              << " newly created stages.";
          return;
        }
        CHECK(curr_stage_name == input_stage_name)
            << "Checking: analyzing stage " << input_stage_name << " but "
            << "got invalid attr stmt op in its body " << op->node;

        string loop_level =
            (for_loop_level > 0)
                ? " (loop level " + std::to_string(for_loop_level) + ")"
                : "";
        HCL_DEBUG_LEVEL(2) << "Stage " << child_stage_name << " attaching to "
                           << curr_stage_name << loop_level << "...";

        // Create a BB entry for current stage
        if (for_loop_level == 0) {
          BlockElem bb;
          bb.name = child_stage_name;
          bb.is_attach_point = true;
          stage_to_attach_children_[curr_stage_name].push_back(bb);
          // Add a loop level to current BB
        } else {
          CHECK_GT(stage_to_attach_children_[curr_stage_name].size(), 0);
          auto& bb = stage_to_attach_children_[curr_stage_name].back();
          CHECK(bb.is_nested_loops);
          bb.attach_point_loop_level = for_loop_level;
        }
      }
    }
    this->Visit(op->body);
  }

  // The stage body does not have ProducerConsumer or Block instance
  // So here we collaspe the most common block (i.e. For and IfThenElse)
  // to represent a computing block in a stage body
  // This information will be used to determine whether a whole stage should
  // be moved to device scope or not
  void Visit_(const For* op) {
    // Create a BB for the netsed for loops
    if (for_loop_level == 0) {
      BlockElem bb;
      bb.name = "for";
      bb.is_nested_loops = true;
      stage_to_attach_children_[input_stage_name].push_back(bb);
    }

    for_loop_level += 1;
    this->Visit(op->body);
    for_loop_level = 0;
    return;
  }

  void VisitStageBody(Stmt stmt, string stage_name) {
    input_stage_name = stage_name;
    this->Visit(stmt);
  }

  unordered_map<string, string>& stage_to_attach_parent_;
  unordered_map<string, vector<BlockElem>>& stage_to_attach_children_;
  string input_stage_name;
  int for_loop_level{0};
};

// Input vector<AttrStmt>
Stmt MakeNestAttr(vector<Stmt>& stack) {
  Stmt body = Evaluate::make(0);
  for (int k = stack.size() - 1; k >= 0; k--) {
    Stmt s = stack[k];
    const AttrStmt* op = s.as<AttrStmt>();
    CHECK(op);
    body = AttrStmt::make(op->node, op->attr_key, op->value, body);
  }
  return body;
}

// Create dfs post ordered attch attr stmt
// The insert_point is used to determine the index of the op
// array from where the subgraph op starts
Stmt AttachScopeReorder(
    int insert_point, vector<Operation>& subgraph,
    vector<Operation>& non_subgraph_ops, vector<Operation>& merged_ops,
    unordered_map<string, string>& stage_to_attach_parent,
    unordered_map<string, vector<BlockElem>>& stage_to_attach_children) {
  Stmt body = Evaluate::make(0);
  Stmt no_op = Evaluate::make(0);
  CHECK_GT(subgraph.size(), 0);

  CHECK(stage_to_attach_children.count("_top"));
  auto second_level_stages = stage_to_attach_children["_top"];
  HCL_DEBUG_LEVEL(2)
      << "============== Top stage children stages before =============";
  for (auto& stage_name : second_level_stages) {
    HCL_DEBUG_LEVEL(2) << stage_name.name;
  }

  // Insert the non-placeholder stages until
  // hitting the threshold index (Since placeholder op
  // does not attach to and parent stage).
  vector<Stmt> stack;
  HCL_DEBUG_LEVEL(2)
      << "============== Top stage children stages after .to() =============";
  int current_index = 0;
  bool aggregated_ops_inserted = false;
  for (auto& op : non_subgraph_ops) {
    if (op->name == "_top") continue;
    if (auto extern_op = op.as<ExternOpNode>()) {
      // Do not create attach points for stages that have an attaching points
      // already Notice that placeholder stage does not have attaching point
      CHECK(stage_to_attach_parent.count(op->name)) << op->name;
      if (stage_to_attach_parent[op->name] != "_top") {
        HCL_DEBUG_LEVEL(2) << "Stage " << op->name
                           << " already has an attaching point...";
        continue;
      }

      // Insert extern op until it hits the insert threshold
      if (current_index == insert_point) {
        aggregated_ops_inserted = true;
        for (auto& m_op : merged_ops) {
          auto m_extern_op = m_op.as<ExternOpNode>();
          Buffer m_buf = m_extern_op->output_placeholders[0];
          HCL_DEBUG_LEVEL(2) << m_buf->data;
          Stmt s = AttrStmt::make(VarExpr(m_buf.node_), attr::attach_scope,
                                  StringImm::make("_top"), no_op);
          stack.push_back(s);
        }
      }

      Buffer buf = extern_op->output_placeholders[0];
      HCL_DEBUG_LEVEL(2) << buf->data;
      Stmt s = AttrStmt::make(VarExpr(buf.node_), attr::attach_scope,
                              StringImm::make("_top"), no_op);
      stack.push_back(s);
    }
    current_index++;
  }

  // If there is no other ExternOp stage after aggregated op
  if (!aggregated_ops_inserted) {
    CHECK(current_index == insert_point);
    HCL_DEBUG_LEVEL(2) << "Inserting aggregated ops at the end...";
    for (auto& m_op : merged_ops) {
      auto m_extern_op = m_op.as<ExternOpNode>();
      Buffer m_buf = m_extern_op->output_placeholders[0];
      HCL_DEBUG_LEVEL(2) << m_buf->data;
      Stmt s = AttrStmt::make(VarExpr(m_buf.node_), attr::attach_scope,
                              StringImm::make("_top"), no_op);
      stack.push_back(s);
    }
  }

  body = MakeNestAttr(stack);
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

// Modify the subgraph and non-subgraph ops
// if any super stage is offloaded to device (i.e. subgraph)
void SubStageOpReorder(
    vector<Operation>& subgraph_ops, vector<Operation>& non_subgraph_ops,
    unordered_map<string, vector<string>> stage_to_substage_on_dev,
    unordered_map<string, vector<BlockElem>>& stage_to_attach_children) {
  HCL_DEBUG_LEVEL(2) << "Reordering the op array if necessary...";
  for (auto& kv : stage_to_substage_on_dev) {
    string super_stage_name = kv.first;
    CHECK(stage_to_attach_children.count(super_stage_name));
    if (stage_to_attach_children[super_stage_name].size() == kv.second.size()) {
      // These super stage should be inserted right before the
      HCL_DEBUG_LEVEL(2) << "Found super stage " << super_stage_name
                         << " fully offloaded to device scope. "
                         << "Remove it from the non-subgraph op array...";
      for (unsigned int i = 0; i < non_subgraph_ops.size(); i++) {
        if (non_subgraph_ops[i]->name == super_stage_name) {
          Operation super_stage_op = non_subgraph_ops[i];
          subgraph_ops.push_back(super_stage_op);
          non_subgraph_ops.erase(non_subgraph_ops.begin() + i);
          break;
        }
      }
    }
  }
}

// How to match pairs of (inputs, outputs) in the graph
// create an aggregate super stage (in merged op)
vector<Operation> ExtractSubGraph(
    const Array<Operation>& roots, const ReadGraph& g, const Schedule& sch,
    unordered_map<const Node*, DevScope>& dev, vector<Operation>& boundary,
    vector<Operation>& merged_ops, vector<Operation>& non_subgraph_ops,
    unordered_map<string, string>& stage_to_attach_parent,
    unordered_map<string, vector<BlockElem>>& stage_to_attach_children,
    bool& schedule_roll_back) {
  // Debug: print the read graph
  if (debug) {
    HCL_DEBUG_LEVEL(2) << "------------ Read Graph -------------";
    for (auto& kv : g) {
      HCL_DEBUG_LEVEL(2) << "------------";
      HCL_DEBUG_LEVEL(2) << "Stage " << kv.first << " reads from ";
      for (auto& t : kv.second) HCL_DEBUG_LEVEL(2) << "    " << t->op;
    }
  }

  Array<Array<Tensor>> inputs, outputs;
  vector<Operation> workset;
  if (boundary.size() == 0) return workset;

  // Set up the search boundary
  for (auto op : boundary) {
    HCL_DEBUG_LEVEL(2) << "Insert boundary op " << op->name << "...";
    workset.insert(workset.begin(), op);
  }

  // The endpoint stages correspond to the tensor to be moved
  Array<Operation> input_ops;
  Array<Operation> output_ops;
  Array<Operation> endpoints;

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
        HCL_DEBUG_LEVEL(2) << "Identify input tensor " << v->name
                           << " of root tensor " << op->name;
        workset.erase(it);
        input.push_back(v.output(0));
        input_ops.push_back(v);
      }
    }
    inputs.push_back(input);
    outputs.push_back(output);

    // If there is only one stage in the boundary set
    // then itself is the only stage in output, and input is empty
    // I.e. Itself is the only stage in subgraph
    if (boundary.size() == 1) break;
    if (input.size() == 0) {
      schedule_roll_back = true;
      LOG(CLEAN)
          << "[ Critical Warning ] Cannot found the subgraph output " << output
          << ". The compilation flow requires the device scope to"
          << " form an enclosed subgraph. Offload the whole program to FPGA...";
      return vector<Operation>();
    }
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
  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();

    // Save op into the subgraph
    HCL_DEBUG_LEVEL(2) << "Add op " << op << " to the subgraph...";
    subgraph.insert(subgraph.begin(), op);
    for (const auto& t : g.at(op)) {
      // Check whether op and its predecessors
      // are in the graph or not. If the
      bool reach_bound = false;
      CHECK(dev.count(t->op.get()));
      if (dev[t->op.get()].is_endpoint) {
        endpoints.push_back(t->op);
        HCL_DEBUG_LEVEL(2) << "    Endpoint found " << t
                           << "... Setup reach_bonud";
        reach_bound = true;
      }

      if (t->op.defined()) {
        if (visited.count(t->op.get()) == 0) {
          visited.insert(t->op.get());

          if (!reach_bound) {
            stack.push_back(t->op);
          }
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
  Buffer aggregate_buffer =
      BufferNode::make(Var(aggregate->name, Handle()), Int(32), Array<Expr>(),
                       Array<Expr>(), Expr(), aggregate->name, "", 0, 0);
  aggregate->output_placeholders.push_back(aggregate_buffer);

  // Create an attach point for all the in-subgraph ops
  // To preserve the order that is implicitly enforced
  vector<string> order_subgraph_ops;
  unordered_set<string> subgraph_op_names;
  for (auto& op : subgraph) {
    string name = op->name;
    CHECK(!subgraph_op_names.count(name))
        << "Error: found duplicate stage name " << name << "...";
    subgraph_op_names.insert(name);
  }
  unordered_map<string, Operation> name2op;
  for (Stage stage : sch->stages) {
    string op_name = stage->op->name;
    CHECK(!name2op.count(op_name)) << "Found stage name duplicate: " << op_name
                                   << " in stages: " << sch->stages;
    name2op[op_name] = stage->op;
    if (subgraph_op_names.count(op_name)) {
      order_subgraph_ops.push_back(op_name);
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
  unordered_map<string, vector<string>> stage_to_substage_on_dev;
  unordered_set<string> attached_stages_record;
  for (int i = reordered_subgraph.size() - 1; i >= 0; i--) {
    auto op = reordered_subgraph[i];

    if (op.as<ExternOpNode>() == NULL) {
      schedule_roll_back = true;
      LOG(CLEAN)
          << "[ Critical Warning ] The graph information is not complete. "
          << " Found placeholder " << op << " in the extracted subgraph. "
          << " Offload the whole program to FPGA...";
      return vector<Operation>();
    }
    auto extern_op = op.as<ExternOpNode>();

    // Check if the op already has an attaching scope
    // If the stage is part of hand-written hcl.Stage,
    // then we will analyze its position in that super stage
    // 1. If it is the last stage of the super stage,
    //    then the whole super stage should be put into the subgraph
    // 2. Part of the super stage (yet to be supported)
    // 3. Other case (e.g. partitioned or reuse): pass
    if (stage_to_attach_parent.count(extern_op->name)) {
      if (stage_to_attach_parent[extern_op->name] != "_top") {
        string parent_stage_name = stage_to_attach_parent[extern_op->name];

        auto set = stage_to_attach_children[parent_stage_name];
        int index = -1;
        // Check this subgraph op's position in its parnet stage body
        for (unsigned int k = 0; k < set.size(); k++) {
          auto& bb = set[k];
          if (bb.name == extern_op->name) index = k;
        }

        auto& substages = stage_to_substage_on_dev[parent_stage_name];
        substages.insert(substages.begin(), extern_op->name);
        HCL_DEBUG_LEVEL(2) << "---- the stage " << extern_op->name
                           << " has a (non-top) parent stage "
                           << parent_stage_name << " (" << index << "/"
                           << set.size() - 1 << "/" << substages.size() << ")";

        // When all its substages are in dev scope
        if (substages.size() == set.size()) {
          // Insert that super stage and notify the top stage that
          // this super stage should be dettached from its body
          // TODO(Hecmay): right now we only support offloading the whole stage
          // that was originally attached to top stage body
          if (stage_to_attach_parent[parent_stage_name] == "_top" &&
              attached_stages_record.find(parent_stage_name) ==
                  attached_stages_record.end()) {
            HCL_DEBUG_LEVEL(2)
                << "INFO : The attaching point of " << extern_op->name
                << " is the last BB in the stage body of " << parent_stage_name
                << ". Move the whole super stage into device scope...";
            attached_stages_record.insert(parent_stage_name);
            CHECK(name2op.count(parent_stage_name));
            auto super_stage_op = name2op[parent_stage_name].as<ExternOpNode>();
            CHECK_GT(super_stage_op->output_placeholders.size(), 0);
            Buffer out_buf = super_stage_op->output_placeholders[0];
            body = AttrStmt::make(VarExpr(out_buf.node_), "attach_scope",
                                  StringImm::make("test"), body);
          }
        }
        continue;
      }
    }

    // If the stage has been attached as a super stage before
    if (attached_stages_record.find(extern_op->name) !=
        attached_stages_record.end())
      continue;
    HCL_DEBUG_LEVEL(2) << "---- Attaching " << extern_op->name;
    attached_stages_record.insert(extern_op->name);
    CHECK_GT(extern_op->output_placeholders.size(), 0);
    Buffer out_buf = extern_op->output_placeholders[0];
    body = AttrStmt::make(VarExpr(out_buf.node_), "attach_scope",
                          StringImm::make("test"), body);
  }

  CHECK_GT(output_ops.size(), 0);
  CHECK(dev.count(output_ops[0].get()));

  // Reorder the subgraph op and non-subgraph-ops with stage offloading in mind
  // 1. If a super stage if offloaded to device
  // 2. If only part of a super stage is offloaded to device
  // 3. Other cases, keep the op array unchanged
  SubStageOpReorder(reordered_subgraph, non_subgraph_ops,
                    stage_to_substage_on_dev, stage_to_attach_children);

  // Collect all the endpoints in the graph
  // The output_ops are endpoints, while the input_ops'
  // parent stages are endpoints (The endpoints are the last stages
  // before scoep change in the graph. The input_ops and output_ops
  // are the boundary of the subgraph)
  for (auto& op : output_ops) {
    auto info = dev[op.get()];
    if (!info.is_endpoint) {
      schedule_roll_back = true;
      LOG(WARNING) << op->name
                   << " should be set as an endpoint... rolling back";
      return vector<Operation>();
    }
    endpoints.push_back(op);
  }

  // Decorate body with attr stmt
  // Push the information stored in endpoint stage
  // into the AttrStmt with attr::io_interface
  for (auto& op : endpoints) {
    auto info = dev[op.get()];
    CHECK(info.is_endpoint) << op->name << " Must be set as an endpoint";
    std::string encode = "";
    encode += std::to_string(static_cast<int>(info.dev_type));
    encode += ":" + std::to_string(static_cast<int>(info.storage_type));
    encode += ":" + std::to_string(info.mem_port);
    encode += ":" + std::to_string(static_cast<int>(info.stream_type));
    encode += ":" + std::to_string(info.channel_depth);
    encode += ":" + std::to_string(info.burst_len);
    VarExpr var(info.target_tensor);
    body =
        AttrStmt::make(var, attr::io_interface, StringImm::make(encode), body);
  }

  Expr scope = StringImm::make("fpga");
  body = AttrStmt::make(VarExpr(), attr::device_scope, scope, body);
  aggregate->body = body;

  if (debug) {
    HCL_DEBUG_LEVEL(2) << "----------- Ops in Subgraph --------------";
    for (auto op : reordered_subgraph) HCL_DEBUG_LEVEL(2) << "    " << op;
    HCL_DEBUG_LEVEL(2)
        << "----------- Aggregate Subgraph Op Body --------------";
    HCL_DEBUG_LEVEL(2) << "\n" << aggregate->body;
  }

  merged_ops.push_back(Operation(aggregate));
  return reordered_subgraph;
}

// Analyze the existing sch->ops and readgraph
// Extracting the graph hierarchy
Array<Operation> HostDevPartition(const Array<Operation>& roots,
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
  unordered_map<string, vector<BlockElem>> stage_to_attach_children;
  AttachingStagesUpdater updater(stage_to_attach_parent,
                                 stage_to_attach_children);

  for (Stage stage : sch->stages) {
    if (auto extern_op = stage->op.as<ExternOpNode>()) {
      // Visit stage op body to collect parent-child information
      updater.VisitStageBody(extern_op->body, stage->op->name);
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
      HCL_DEBUG_LEVEL(2) << "Endpoint stage " << stage;
      info.is_endpoint = true;
      info.storage_type = stage->endpoint.storage_type;
      info.stream_type = stage->endpoint.stream_type;
      info.mem_port = stage->endpoint.mem_port;
      info.channel_depth = stage->endpoint.channel_depth;
      info.burst_len = stage->endpoint.burst_len;
      info.target_tensor = stage->endpoint.target_tensor;
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
  bool schedule_roll_back = false;
  vector<Operation> subgraph = ExtractSubGraph(
      roots, g, sch, dev, boundary, merged_ops, non_subgraph_ops,
      stage_to_attach_parent, stage_to_attach_children, schedule_roll_back);

  // If we failed to extract the subgraph, then automatically offload the whole
  // DFG to the FPGA scope
  if (schedule_roll_back) {
    // Create a new op that has the top stage as child
    auto ret_ops = PostDFSOrder(roots, g);

    size_t s = ret_ops.size() - 1;
    auto top_op = ret_ops[s];
    CHECK(top_op->name == "_top") << top_op->name;

    std::shared_ptr<ExternOpNode> new_op = std::make_shared<ExternOpNode>();
    new_op->name = "__device_scope";
    auto extern_op = top_op.as<ExternOpNode>();

    new_op->inputs = std::move(extern_op->inputs);
    new_op->input_placeholders = std::move(extern_op->input_placeholders);

    Buffer void_buffer = BufferNode::make(Var("__device_scope", Handle()),
                                          Int(32), Array<Expr>(), Array<Expr>(),
                                          Expr(), "__device_scope", "", 0, 0);
    new_op->output_placeholders.push_back(void_buffer);

    Buffer buf = extern_op->output_placeholders[0];
    HCL_DEBUG_LEVEL(2) << buf->data;
    Stmt no_op = Evaluate::make(0);
    Stmt body = AttrStmt::make(VarExpr(buf.node_), attr::attach_scope,
                               StringImm::make("__device_scope"), no_op);

    Expr scope = StringImm::make("fpga");
    body = AttrStmt::make(VarExpr(), attr::device_scope, scope, body);
    // Get endpoint information
    for (auto& op : ret_ops) {
      CHECK(dev.count(op.get())) << op;
      if (dev[op.get()].is_endpoint) {
        HCL_DEBUG_LEVEL(2)
            << "[ info ] found incomplete placement information. "
            << "Save the op " << op << " info into the IR...";
        auto info = dev[op.get()];
        std::string encode = "";
        encode += std::to_string(static_cast<int>(info.dev_type));
        encode += ":" + std::to_string(static_cast<int>(info.storage_type));
        encode += ":" + std::to_string(info.mem_port);
        encode += ":" + std::to_string(static_cast<int>(info.stream_type));
        encode += ":" + std::to_string(info.channel_depth);
        encode += ":" + std::to_string(info.burst_len);

        VarExpr var(info.target_tensor);
        body = AttrStmt::make(var, attr::io_interface, StringImm::make(encode),
                              body);
      }

      // Check the unattached partition stages for placeholders
      if (!stage_to_attach_parent.count(op->name)) {
        if (op->name != "_top" && op.as<ExternOpNode>() &&
            op->name.find("partitioned") != std::string::npos) {
          HCL_DEBUG_LEVEL(2)
              << "[ debug ] found stage " << op->name << " attached nowhere. "
              << "Create an attaching point for it...";
          auto extern_op = op.as<ExternOpNode>();
          Buffer buf = extern_op->output_placeholders[0];
          body = AttrStmt::make(VarExpr(buf.node_), attr::attach_scope,
                                StringImm::make("__device_scope"), body);
        }
      }
    }

    new_op->body = body;
    auto new_top = Operation(new_op);
    HCL_DEBUG_LEVEL(2) << "[ debug ] new top level body: \n " << body;

    ret_ops.push_back(new_top);
    return ret_ops;
  }

  // Insert the subgraph ops and the super stage op. Also update
  // the top stage body in different cases
  //   1. The subgraph ops are not part of any super stages
  //   2. The subgraph contains some super stages
  //   3. The subgraph has intersection with other super stages (Not supported)
  if (merged_ops.size() > 0) {
    vector<Operation> results;

    // First insert all the endpoint stages
    // Insert the new aggregate op after last endpoint stage in the op array
    // The insert point is also used for yop stage body recobstruction
    int insert_point = 0;
    for (size_t m = 0; m < non_subgraph_ops.size(); m++) {
      auto op = non_subgraph_ops[m];
      if (dev[op.get()].is_endpoint) {
        insert_point = m + 1;
        HCL_DEBUG_LEVEL(2) << "Ops (ep) outside the subgraph : "
                           << non_subgraph_ops[m];
      } else {
        HCL_DEBUG_LEVEL(2) << "Ops outside the subgraph : "
                           << non_subgraph_ops[m];
      }
      results.push_back(op);
    }

    // Check the attaching substage before the insert point
    // E.g. Placeholder(A), A.reuse, extern(B, endpoint), extern(C)
    // Here the A.reuse is attached to extern(B), so the actual insert point
    // (to insert merged_op into top stage bodt as a attaching point) whould be
    // insert_point - 1 (i.e. the A.reuse, which is the num of atatching
    // substage before the insert_point)
    int host_non_top_substage_num = 0;
    for (int k = 0; k < insert_point; k++) {
      auto op = non_subgraph_ops[k];
      auto name = op->name;
      if (stage_to_attach_parent.count(name)) {
        if (stage_to_attach_parent.at(name) != "_top") {
          HCL_DEBUG_LEVEL(2) << "--- Found non-top attaching substage " << name
                             << " beofore insert point...";
          host_non_top_substage_num++;
        }
      }
    }

    HCL_DEBUG_LEVEL(2) << "INFO: Final insert point index : " << insert_point;
    int current_index = 0;
    for (size_t k = 0; k < subgraph.size(); k++) {
      Operation op = subgraph[k];
      HCL_DEBUG_LEVEL(2) << "Ops in the subgraph : " << op;
      results.insert(results.begin() + insert_point + current_index, op);
      current_index++;
    }

    // Insert the merged op
    for (auto& op : merged_ops) {
      int index = insert_point + subgraph.size();
      results.insert(results.begin() + index, op);
    }

    // Rearrange attachment scope attr inside _top body
    // We should remove the original attach points in _top stage body
    // and insert the merged ops instead
    Array<Operation> post_order;
    for (auto& op : results) {
      if (op->name == "_top") {
        Stmt no_op = Evaluate::make(0);
        std::shared_ptr<ExternOpNode> new_op = std::make_shared<ExternOpNode>();
        new_op->name = op->name;
        auto extern_op = op.as<ExternOpNode>();
        CHECK(extern_op) << "The top op node is not an ExternOpNode...";

        HCL_DEBUG_LEVEL(2) << "========= Original Top Op Body =============";
        HCL_DEBUG_LEVEL(2) << extern_op->body;

        new_op->inputs = std::move(extern_op->inputs);
        new_op->input_placeholders = std::move(extern_op->input_placeholders);
        new_op->output_placeholders = std::move(extern_op->output_placeholders);
        auto attr_insert_point = insert_point - host_non_top_substage_num;
        new_op->body = AttachScopeReorder(
            attr_insert_point, subgraph, non_subgraph_ops, merged_ops,
            stage_to_attach_parent, stage_to_attach_children);

        HCL_DEBUG_LEVEL(2) << "========= Restrctured Top Op Body =============";
        HCL_DEBUG_LEVEL(2) << new_op->body;
        op = Operation(new_op);
      }
      post_order.push_back(op);
    }

    CHECK(results.size() == sch->stages.size() + merged_ops.size())
        << "Missing ops in result. size " << results.size() << ":"
        << sch->stages.size() << post_order;

    HCL_DEBUG_LEVEL(2) << "\n";
    HCL_DEBUG_LEVEL(2)
        << "=========== Ops after schedule reorder =============";
    HCL_DEBUG_LEVEL(2) << post_order;
    HCL_DEBUG_LEVEL(2) << "\n";
    HCL_DEBUG_LEVEL(2)
        << "=========== Ops before schedule reorder =============";
    HCL_DEBUG_LEVEL(2) << PostDFSOrder(roots, g) << "\n";
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
  if (sch->super_stages.size() > 0) {
    HCL_DEBUG_LEVEL(2) << "Already partitioned scope."
                       << " return original schedule...";
    return sch;
  }

  // Extract the root ops of CDFG
  Array<Operation> roots;
  for (Operation op : sch->outputs) {
    if (sch->stage_map[op]->is_output) roots.push_back(sch->stage_map[op]->op);
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
        CHECK(sch->stage_map.count(t->op)) << "cannot find " << t->op;
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
    std::shared_ptr<StageNode> snode =
        std::make_shared<StageNode>(*stage.operator->());
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
  for (Operation op : post_order) {
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
          // Replace stage op body for _top
          if (scopy->op->name == "_top") {
            scopy = Stage(op);
            n->stage_map.Set(op, scopy);
          }

          HCL_DEBUG_LEVEL(2)
              << "---- Adding stage copy " << scopy->op->name << "...";
          n->stages.push_back(scopy);
        }
      }
    }

    // Merged stage hypernode
    if (!scopy.defined()) {
      HCL_DEBUG_LEVEL(2) << "---- Creating merged stage " << op->name << "...";
      Stage stage = Stage(op);
      n->stage_map.Set(op, stage);
      n->stages.push_back(stage);
      n->super_stages.push_back(stage);
    }
    // FIXME: stage_map op inconsistent with s->op
    // CHECK(sch->stage_map.count(op));
  }

  for (Stage g : sch->groups) {
    std::shared_ptr<StageNode> gnode =
        std::make_shared<StageNode>(*g.operator->());
    Stage gcopy = Stage(gnode);
    smap[g] = gcopy;
    n->groups.push_back(gcopy);
  }

  // Remap ops to new stages
  for (auto kv : sch->stage_map) {
    if (smap.count(kv.second)) n->stage_map.Set(kv.first, smap.at(kv.second));
  }

  for (Stage s : n->stages) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found";
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end()) << s->group << " not found";
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
      CHECK(smap.find(s->group) != smap.end()) << s->group << " not found";
      s->group = smap.at(s->group);
    }
  }
  return Schedule(n);
}

}  // namespace schedule
}  // namespace TVM
