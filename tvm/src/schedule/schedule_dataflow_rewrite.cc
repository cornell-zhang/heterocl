/*!
 *  Copyright (c) 2020 by Contributors
 * \file schedule_dataflow_rewrite.cc
 */
#include <arithmetic/Substitute.h>
#include <tvm/buffer.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule.h>
#include <regex>
#include <unordered_set>
#include "../arithmetic/compute_expr.h"
#include "../pass/ir_util.h"
#include "./message_passing.h"

namespace TVM {

using namespace ir;

// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Node* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

// replace in stage expr & stmt
class InStageMover : public ir::IRMutator {
 public:
  explicit InStageMover(const Expr& scope, const int index)
      : scope_{scope}, index_(index) {}

  Stmt Mutate_(const For* op, const Stmt& s) {
    if (counter == index_) {
      return AttrStmt::make(VarExpr(), attr::device_scope, scope_, s);
    } else {
      counter += 1;
      return For::make(op->loop_var, op->min, op->extent, op->for_type,
                       op->device_api, this->Mutate(op->body),
                       op->annotate_keys, op->annotate_values);
    }
  }

 private:
  const Expr& scope_;
  const int index_;
  int counter{0};
};

// The replacer of cache.
class VarReplacer : public ir::IRMutator {
 public:
  explicit VarReplacer(const std::unordered_map<const Variable*, Expr>& vsub)
      : vsub_(vsub) {}

  Expr Mutate_(const Variable* op, const Expr& e) {
    auto it = vsub_.find(op);
    if (it != vsub_.end()) return it->second;
    return e;
  }

 private:
  const std::unordered_map<const Variable*, Expr>& vsub_;
};

class ParentStmtCollector final : public IRMutator {
 public:
  ParentStmtCollector(const VarExpr& target_buf, const VarExpr& reuse_buf,
                      const std::string& parent_name, const IterVar& axis)
      : target_buf_(target_buf),
        reuse_buf_(reuse_buf),
        parent_name_(parent_name),
        axis_(axis) {
    CHECK(target_buf.defined());
  }

  Stmt Mutate_(const For* op, const Stmt& s) {
    if (op->loop_var.get() == axis_->var.get()) {
      const AttrStmt* attr = op->body.as<AttrStmt>();
      Stmt attr_stmt =
          AttrStmt::make(reuse_buf_, "attach_scope",
                         StringImm::make(parent_name_), attr->body);
      attr_stmt =
          AttrStmt::make(attr->node, attr->attr_key, attr->value, attr_stmt);
      Stmt reuse_stmt = Reuse::make(target_buf_, attr_stmt);
      return For::make(op->loop_var, op->min, op->extent, op->for_type,
                       op->device_api, reuse_stmt, op->annotate_keys,
                       op->annotate_values);
    } else {
      return For::make(op->loop_var, op->min, op->extent, op->for_type,
                       op->device_api, IRMutator::Mutate(op->body),
                       op->annotate_keys, op->annotate_values);
    }
  }

 private:
  const VarExpr& target_buf_;
  const VarExpr& reuse_buf_;
  const std::string& parent_name_;
  const IterVar& axis_;
};

Expr InjectPredicate(const Array<Expr>& predicates, Expr body) {
  using ir::Reduce;
  using ir::Select;
  if (predicates.size() == 0) return body;
  const Reduce* reduce = body.as<Reduce>();
  if (reduce) {
    std::shared_ptr<Reduce> n = std::make_shared<Reduce>(*reduce);
    n->condition =
        n->condition && arith::ComputeReduce<ir::And>(predicates, Expr());
    return Expr(n);
  }
  return Select::make(arith::ComputeReduce<ir::And>(predicates, Expr()), body,
                      make_zero(body.type()));
}

// Replace data flow appears in all stages given the tensor change.
// Also update vmap if subsequent dataflow need to be replaced.
void ReplaceDataFlow(const Array<Stage>& stages,
                     std::unordered_map<Tensor, Tensor>* vmap) {
  for (Stage s : stages) {
    Operation op = s->op->ReplaceInputs(s->op, *vmap);
    if (!op.same_as(s->op)) {
      for (int i = 0; i < op->num_outputs(); ++i) {
        (*vmap)[s->op.output(i)] = op.output(i);
      }
      s->op = op;
    }
  }
}

// IRMutator used for inter-module streaming
// Only used to inject information into target KernelDef
class InfoUpdater final : public IRMutator {
 public:
  static int channelCount;
  InfoUpdater(const int arg_pos, const int channel_depth,
              const int channel_index, const bool is_sender)
      : arg_pos_(arg_pos),
        channel_depth_(channel_depth),
        channel_index_(channel_index),
        is_sender_(is_sender) {}

  // Add information into KernelDef as AttrStmt
  Stmt Mutate_(const KernelDef* op, const Stmt& s) {
    Array<Array<Expr>> arr = op->attributes;
    CHECK(op->attributes.size() <= op->args.size());

    auto name = op->args[arg_pos_].get()->name_hint;
    std::string info = std::to_string(arg_pos_);
    info += ":" + std::to_string(channel_index_);
    info += ":" + std::to_string(channel_depth_);
    info += ":" + std::string(is_sender_ ? "1" : "0");

    // Substitute load inside kernel def to stream channels
    VarExpr node(op->args[arg_pos_].node_);
    Stmt body =
        AttrStmt::make(node, "kernel_stream", StringImm::make(info), op->body);

    return KernelDef::make(op->args, op->arg_shapes, op->arg_types,
                           op->arg_tensors, body, op->ret_void, op->ret_type,
                           op->name, op->attributes);
  }

 private:
  const int arg_pos_;
  const int channel_depth_;
  int channel_index_{0};
  const bool is_sender_;
};

// Initialize static channel count
int InfoUpdater::channelCount = 0;

// stream buffer data to kernel stage
void Schedule::to_stage(const Tensor& target,
                        /*kernel def stage*/ Stage dest,
                        /*position index*/ int arg_pos, StreamType stream_type,
                        int channel_depth, std::string name) {
  Stage target_stage = (*this)[target];
  Buffer target_buffer;

  // target stage as kernel def operator (receiver)
  if (auto op = target_stage->op.as<ExternOpNode>()) {
    target_buffer = op->output_placeholders[0];
    Buffer input_buffer = op->input_placeholders[0];

    // mutate to add new sender
    target_stage->op = ExternOpNode::make(op->name, "", Array<IterVar>(),
                                          op->inputs, op->input_placeholders,
                                          op->output_placeholders, op->body);
    // update dest stage body for data stream in
    const ExternOpNode* destOp = dest->op.as<ExternOpNode>();
    dest->op = ExternOpNode::make(destOp->name, destOp->tag, destOp->axis,
                                  destOp->inputs, destOp->input_placeholders,
                                  Array<Buffer>(), destOp->body);
  }
}

// Stream data between hardware modules
void Schedule::stream_to(const Tensor& target, Stage dest, Stage source,
                         Array<Expr> stream_pos, StreamType stream_type,
                         int channel_depth, Array<IterVar> axis) {
  Stage target_stage = (*this)[target];
  std::vector<Stage> consumers;
  size_t num_stage = (*this)->stages.size();
  const ExternOpNode* destOp = dest->op.as<ExternOpNode>();
  const ExternOpNode* srcOp = source->op.as<ExternOpNode>();

  // Extract target buffer and consumer stages of the channel.
  // When a global buffer is streamed between modules,
  // the target tensor is supposed to modified by only two stages (one
  // is the consumer and another is the producer)
  // Example:
  //     allocate buffer;
  //     function_one(buffer); // write to buffer (stage one)
  //     function_two(buffer); // read from buffer (stage two)
  const ExternOpNode* op = target_stage->op.as<ExternOpNode>();
  if (op == NULL) {
    LOG(CLEAN) << "Target tensor " << target << " is placeholder "
               << "and cannot be streamed to an on-chip consumer...";
    LOG(CLEAN) << "Consider using s.to(" << target->op->name
               << ", target.xcel)...";
    return;
  }

  Buffer target_buffer = op->output_placeholders[0];
  consumers.push_back(target_stage);
  for (size_t i = 0; i < num_stage; i++) {
    Stage s = (*this)->stages[i];
    if (const ExternOpNode* op = s->op.as<ExternOpNode>()) {
      for (size_t j = 0; j < op->inputs.size(); j++) {
        if (target_buffer == op->input_placeholders[j]) {
          consumers.push_back(s);
        }
      }
    }
  }

  // Inter-stage data movement
  if (stream_pos.size() == 0) {
    // Self loop-back case
    if (destOp == srcOp) {
      // Annotate the allocate node
      HCL_DEBUG_LEVEL(2) << "[ debug ] Streaming tensor " << target_buffer
                         << " to stage " << destOp->name << " (loopback)...";
      VarExpr node(target_buffer->data.node_);
      Stmt dest_body =
          AttrStmt::make(node, attr::device_scope,
                         IntImm::make(Int(32), channel_depth), destOp->body);
      dest->op = ExternOpNode::make(destOp->name, destOp->tag, destOp->axis,
                                    destOp->inputs, destOp->input_placeholders,
                                    destOp->output_placeholders, dest_body);

      // To create stage-to-stage channel.
    } else {
      bool create_stream_array = false;
      if (axis.size() > 0) {
        CHECK_EQ(axis.size(), 2);
        create_stream_array = true;
      }
      VarExpr node(target_buffer->data.node_);
      InfoUpdater::channelCount += 1;
      auto channel_index = InfoUpdater::channelCount;
      int num_of_consumers = 0;
      for (auto s : consumers) {
        if (s->op->name != "_top" && s->op->name != target->op->name) {
          HCL_DEBUG_LEVEL(2) << "Consumer " << s;
          num_of_consumers++;
        }
      }

      if (num_of_consumers > 1) {
        LOG(INFO) << "Tensor " << target->op->name
                  << " has more than one consumers. Start multi-casting...";
      }

      // Create a stream scope for consumer stage
      std::string s = std::to_string(channel_index);
      s += ":" + std::to_string(channel_depth);
      s += ":" + std::to_string(0);
      s += ":" + std::to_string(num_of_consumers);
      Stmt dest_body = AttrStmt::make(node, attr::stream_attrs,
                                      StringImm::make(s), destOp->body);

      dest->op = ExternOpNode::make(destOp->name, destOp->tag, destOp->axis,
                                    destOp->inputs, destOp->input_placeholders,
                                    destOp->output_placeholders, dest_body);

      // Producer stage
      s = std::to_string(channel_index);
      s += ":" + std::to_string(channel_depth);
      s += ":" + std::to_string(1);
      s += ":" + std::to_string(num_of_consumers);
      Stmt src_body = AttrStmt::make(node, attr::stream_attrs,
                                     StringImm::make(s), srcOp->body);
      source->op = ExternOpNode::make(srcOp->name, srcOp->tag, srcOp->axis,
                                      srcOp->inputs, srcOp->input_placeholders,
                                      srcOp->output_placeholders, src_body);
    }

    // Streaming between HCL modules
  } else {
    CHECK(stream_pos.size() == 2) << "Missing pos index";
    int destPos = stream_pos[0].as<IntImm>()->value;
    int srcPos = stream_pos[1].as<IntImm>()->value;

    int num_of_consumers = 0;
    for (auto s : consumers) {
      if (s->op->name != "_top" && s->op->name != target->op->name) {
        HCL_DEBUG_LEVEL(2) << "Consumer " << s;
        num_of_consumers++;
      }
    }
    CHECK(num_of_consumers == 2)
        << "The streaming channel " << target
        << " can only have one producer and one consumer...";

    // Create common channel buffer
    // This is useful for creating global channels
    // E.g. Intel AOC autorun channels
    InfoUpdater::channelCount += 1;
    auto channel_index = InfoUpdater::channelCount;

    // Update annotation in kernek def stmt
    bool dest_stage_is_sender = false;
    bool src_stage_is_sender = true;

    // Inject information to the KernelDef IR node
    // Just add some annotation to avoid potential conflicts
    // If destOp == srcOp, the self-loopback mode will handled later
    InfoUpdater destMutator(destPos, channel_index, channel_depth,
                            dest_stage_is_sender);
    InfoUpdater srcMutator(srcPos, channel_index, channel_depth,
                           src_stage_is_sender);

    HCL_DEBUG_LEVEL(2)
        << "[ INFO ] inter-kernel streaming. create channel index "
        << channel_index << " (depth=" << channel_depth << ")";

    Stmt dest_body = destMutator.Mutate(destOp->body);
    dest->op = ExternOpNode::make(destOp->name, destOp->tag, destOp->axis,
                                  destOp->inputs, destOp->input_placeholders,
                                  destOp->output_placeholders, dest_body);

    Stmt src_body = srcMutator.Mutate(srcOp->body);
    source->op = ExternOpNode::make(srcOp->name, srcOp->tag, srcOp->axis,
                                    srcOp->inputs, srcOp->input_placeholders,
                                    srcOp->output_placeholders, src_body);
    // Insert an attribute statement into the target stage
    VarExpr node(target_buffer->data.node_);
    std::string info = std::to_string(InfoUpdater::channelCount) + ":" +
                       std::to_string(channel_depth);

    // The stream_scope indicates that
    Stmt target_body = AttrStmt::make(node, attr::stream_scope,
                                      StringImm::make(info), op->body);

    target_stage->op = ExternOpNode::make(op->name, op->tag, op->axis,
                                          op->inputs, op->input_placeholders,
                                          op->output_placeholders, target_body);
  }
}

// move substages within HeteroCL stage
void Schedule::stage_move(Stage parent, DeviceType device_type,
                          StreamType stream_type, int channel_depth,
                          int occur_index) {
  Expr scope;
  switch (device_type) {
    case DeviceType::devHost: {
      scope = StringImm::make("cpu");
      break;
    }
    case DeviceType::devFPGA: {
      scope = StringImm::make("fpga");
      break;
    }
  }

  CHECK(scope.defined()) << "unsopport device ";
  const ExternOpNode* op = parent->op.as<ExternOpNode>();
  CHECK(op) << parent << " not a extern op";
  Stmt body = InStageMover(scope, occur_index).Mutate(op->body);

  // result must be moved back before stage ends
  parent->op =
      ExternOpNode::make(op->name, op->tag, op->axis, op->inputs,
                         op->input_placeholders, op->output_placeholders, body);
}

// Move data to device
Tensor Schedule::move_to(const Tensor& target, Stage parent,
                         DeviceType device_type, StreamType stream_type,
                         int channel_depth, Array<Expr> dev_ports) {
  Stage target_stage = (*this)[target];
  std::vector<Stage> consumers;
  size_t num_stage = (*this)->stages.size();
  Buffer target_buffer;

  // parse the memory module interface
  CHECK_EQ(dev_ports.size(), 3);
  auto mem_type = dev_ports[0].as<IntImm>()->value;
  StorageType storage = static_cast<StorageType>(mem_type);

  // Bind IO inertface to on-chip storage
  if (storage == StorageType::devBRAM || storage == StorageType::devLUTRAM ||
      storage == StorageType::devURAM) {
    auto binding = dev_ports[1];
    const ExternOpNode* op = target_stage->op.as<ExternOpNode>();
    if (op == NULL) {
      LOG(WARNING) << "Cannot bind top module port to FPGA on-chip memory...";
      return target;
    }
    target_buffer = op->output_placeholders[0];
    VarExpr node(target_buffer->data.node_);

    Stmt body = AttrStmt::make(node, attr::bind_scope, binding, op->body);
    target_stage->op = ExternOpNode::make(op->name, op->tag, op->axis,
                                          op->inputs, op->input_placeholders,
                                          op->output_placeholders, op->body);
    return target;
  }

  auto mem_port = dev_ports[1].as<IntImm>()->value;
  auto burst_len = dev_ports[2].as<IntImm>()->value;

  // For placeholder typed tensor, we collect all its consumer stages
  // and set these stages in the on-device scope
  if (target_stage->op.as<PlaceholderOpNode>() != nullptr) {
    for (size_t i = 0; i < num_stage; i++) {
      Stage s = (*this)->stages[i];
      if (const ExternOpNode* extern_op = s->op.as<ExternOpNode>()) {
        for (size_t j = 0; j < extern_op->inputs.size(); j++) {
          if (target == extern_op->inputs[j]) {
            target_buffer = extern_op->input_placeholders[j];
            consumers.push_back(s);
            break;
          }
        }
      }
    }

    // The target tensor to be moved is produced
    // by an ExternOp stage.
  } else {
    const ExternOpNode* op = target_stage->op.as<ExternOpNode>();
    target_buffer = op->output_placeholders[0];
    for (size_t i = 0; i < num_stage; i++) {
      Stage s = (*this)->stages[i];
      if (const ExternOpNode* stage_op = s->op.as<ExternOpNode>()) {
        for (size_t j = 0; j < stage_op->inputs.size(); j++) {
          if (op->output_placeholders[0] == stage_op->input_placeholders[j]) {
            consumers.push_back(s);
          }
        }
      }
    }
  }

  // If the parent stage is defined, it means that the target
  // is created in some other stage and updated in this `parent` stage.
  // Otherwise (i.e., `parent` stage is not defined), it means that
  // the target tensor is created instead of being updated
  if (parent.defined()) {
    target_stage = parent;
    const ExternOpNode* op = parent->op.as<ExternOpNode>();
    CHECK(op) << parent << " not a extern op";
    CHECK(target_buffer.defined()) << " not found buffer for target tensor";

    consumers.clear();
    for (size_t i = 0; i < num_stage; i++) {
      Stage s = (*this)->stages[i];
      if (const ExternOpNode* stage_op = s->op.as<ExternOpNode>()) {
        for (size_t j = 0; j < stage_op->inputs.size(); j++) {
          if (op->output_placeholders[0] == stage_op->input_placeholders[j]) {
            consumers.push_back(s);
          }
        }
      }
    }
  }

  // Save the attribute information
  Interface endpoint(storage, stream_type, mem_port, channel_depth, burst_len,
                     target->op->name);
  auto consumers_dev_type = device_type;

  // The stage is the update stage of the target tensor
  // In this case the s->op->output_placeholders does not
  // include the tensor to be updated
  std::string from =
      (parent.defined()) ? (" (updated) from stage " + parent->op->name) : "";
  target_stage->endpoint = endpoint;
  if (device_type == DeviceType::devHost) {
    HCL_DEBUG_LEVEL(2) << "Moving tensor " << target->op->name << from
                       << " to Host...";
    target_stage->device_type = DeviceType::devFPGA;
  } else {
    HCL_DEBUG_LEVEL(2) << "Moving tensor " << target->op->name << from
                       << " to FPGA...";
    target_stage->device_type = DeviceType::devHost;
  }

  // Update consumer stages with new tensor and buffer
  // If a stage is moved to device (host) scope, we consider
  // itself as the endpoint in the CDFG. It is necessary
  // that all of its consumers are in the device (host) scope.
  // Notice: with the flattened CDFG, we shuold not mark a consumer
  //         if it is the parent stage of the target stage
  for (Stage s : consumers) {
    CHECK(s->op.as<ExternOpNode>());
    auto op = s->op.as<ExternOpNode>();

    s->op = ExternOpNode::make(op->name, op->tag, op->axis, op->inputs,
                               op->input_placeholders, op->output_placeholders,
                               op->body);
    std::string scope = (device_type == DeviceType::devHost) ? "Host" : "FPGA";
    if (op->name != "_top") {
      HCL_DEBUG_LEVEL(2) << "Mark stage " << op->name << " on " << scope
                         << " scope...";
      s->device_type = consumers_dev_type;
    }
    (*this)->stage_map.Set(s->op, s);
  }
  return target;
}

Tensor Schedule::reuse_at(const Tensor& target, Stage parent, IterVar axis,
                          std::string reuse_name) {
  const ExternOpNode* op = parent->op.as<ExternOpNode>();
  Array<Tensor> reuse_inputs, new_inputs;
  Array<Buffer> reuse_input_placeholders, new_input_placeholders;
  Array<Buffer> reuse_output_placeholders;
  Stmt new_body;
  // the input is just the target
  Buffer target_buf;
  for (size_t i = 0; i < op->inputs.size(); i++) {
    if (target == op->inputs[i]) {
      reuse_inputs.push_back(target);
      target_buf = op->input_placeholders[i];
      reuse_input_placeholders.push_back(target_buf);
      break;
    }
  }

  // reuse in kernel module
  VarExpr target_var;
  if (!target_buf.defined()) {
    if (auto kernel = op->body.as<KernelDef>()) {
      auto name = target->op->name;
      for (size_t i = 0; i < kernel->args.size(); i++) {
        if (name == kernel->args[i].get()->name_hint) {
          target_var = VarExpr(kernel->args[i].node_);
          break;
        }
      }
    } else {
      LOG(FATAL) << "var " << target << "not found in input buffers";
    }
  } else {  // reuse target buffer varexpr
    target_var = VarExpr(target_buf->data.node_);
  }

  // create an output buffer
  Buffer reuse_output_buf =
      BufferNode::make(Var(reuse_name, Handle()), target->dtype, Array<Expr>(),
                       Array<Expr>(), Expr(), reuse_name, "", 0, 0);
  reuse_output_placeholders.push_back(reuse_output_buf);
  // traverse the parent body and collect the new information
  VarExpr buffer_var = VarExpr(reuse_output_buf.node_);
  ParentStmtCollector mutator(target_var, buffer_var, op->name, axis);
  new_body = mutator.Mutate(op->body);
  // create reuse tensor
  Tensor reuse =
      ExternOpNode::make(reuse_name, "", Array<IterVar>(), reuse_inputs,
                         reuse_input_placeholders, reuse_output_placeholders,
                         Evaluate::make(0))
          .output(0);
  // update parent stage
  new_inputs = op->inputs;
  new_inputs.push_back(reuse);
  new_input_placeholders = op->input_placeholders;
  new_input_placeholders.push_back(reuse_output_buf);
  parent->op = ExternOpNode::make(op->name, op->tag, op->axis, new_inputs,
                                  new_input_placeholders,
                                  op->output_placeholders, new_body);
  // create new stage
  Stage reuse_stage = Stage(reuse->op);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t pos = FindNodeRef(stages, parent);
  stages->data.insert(stages->data.begin() + pos, reuse_stage.node_);
  (*this)->stage_map.Set(reuse->op, reuse_stage);
  return reuse;
}

Tensor Schedule::partition(const Tensor& target, int dim, int factor,
                           PartitionType partition_type) {
  Stage target_stage = (*this)[target];
  std::vector<Stage> consumers;
  size_t num_stage = (*this)->stages.size();
  size_t min_pos = num_stage;
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  Buffer target_buffer;
  const PlaceholderOpNode* op = target_stage->op.as<PlaceholderOpNode>();
  bool is_placeholder = op ? true : false;
  // check if it is a placeholder or not
  if (is_placeholder) {
    min_pos = 0;
    // collect all stages that take the target as inputs
    for (size_t i = 0; i < num_stage; i++) {
      Stage s = (*this)->stages[i];
      if (const ExternOpNode* op = s->op.as<ExternOpNode>()) {
        for (size_t j = 0; j < op->inputs.size(); j++) {
          if (target == op->inputs[j]) {
            target_buffer = op->input_placeholders[j];
            consumers.push_back(s);
            break;
          }
        }
      }
    }
  } else {
    min_pos = FindNodeRef(stages, target_stage);
    const ExternOpNode* op = target_stage->op.as<ExternOpNode>();
    target_buffer = op->output_placeholders[0];
    consumers.push_back(target_stage);
  }
  // build the body of the new stage
  Stmt body = Partition::make(target_buffer->data, dim, factor, partition_type);
  // build the new stage
  Array<Tensor> partition_inputs;
  Array<Buffer> partition_input_placeholders;
  Array<Buffer> partition_output_placeholders;
  std::string partition_name = target_buffer->name + ".partitioned";
  Buffer partition_buffer =
      BufferNode::make(Var(partition_name, Handle()), Int(32), Array<Expr>(),
                       Array<Expr>(), Expr(), partition_name, "", 0, 0);
  if (is_placeholder) {
    partition_inputs.push_back(target);
    partition_input_placeholders.push_back(target_buffer);
  }
  partition_output_placeholders.push_back(partition_buffer);
  Tensor partition_tensor =
      ExternOpNode::make(partition_name, "", Array<IterVar>(), partition_inputs,
                         partition_input_placeholders,
                         partition_output_placeholders, body)
          .output(0);
  Stage partition_stage = Stage(partition_tensor->op);
  stages->data.insert(stages->data.begin() + min_pos, partition_stage.node_);
  (*this)->stage_map.Set(partition_tensor->op, partition_stage);
  // replace the intput of those stages with the new tensor and buffer
  for (size_t i = 0; i < consumers.size(); i++) {
    Stage s = consumers[i];
    Array<Tensor> new_inputs;
    Array<Buffer> new_input_placeholders;
    const ExternOpNode* op = s->op.as<ExternOpNode>();
    new_inputs.push_back(partition_tensor);
    new_input_placeholders.push_back(partition_buffer);
    for (size_t j = 0; j < op->inputs.size(); j++) {
      new_inputs.push_back(op->inputs[j]);
      new_input_placeholders.push_back(op->input_placeholders[j]);
    }
    if (is_placeholder) {
      s->op = ExternOpNode::make(op->name, op->tag, op->axis, new_inputs,
                                 new_input_placeholders,
                                 op->output_placeholders, op->body);
    } else {
      Stmt new_body =
          AttrStmt::make(VarExpr(partition_buffer.node_), "attach_scope",
                         StringImm::make(target_buffer->name), op->body);
      s->op = ExternOpNode::make(op->name, op->tag, op->axis, new_inputs,
                                 new_input_placeholders,
                                 op->output_placeholders, new_body);
    }
  }
  return partition_tensor;
}

// Do not support reshaping the placeholders for now
void Schedule::reshape(const Tensor& target, Array<Expr> new_shape) {
  Stage target_stage = (*this)[target];
  const ExternOpNode* op = target_stage->op.as<ExternOpNode>();
  Buffer target_buffer = op->output_placeholders[0];
  // check the #elem is the same for both shapes
  size_t size = 1, origin = 1;
  for (auto& dim : new_shape) {
    CHECK(dim.as<IntImm>()) << dim << " must be a positive integrer";
    size *= dim.as<IntImm>()->value;
  }
  for (auto& dim : target_buffer->shape) {
    CHECK(dim.as<IntImm>()) << dim << " must be a positive integrer";
    origin *= dim.as<IntImm>()->value;
  }
  CHECK_EQ(origin, size)
      << "new shape must have same element number as original shape";
  target_buffer->shape = new_shape;
}

inline bool ReduceEqual(const ir::Reduce* a, const ir::Reduce* b) {
  return (a->combiner.same_as(b->combiner)) && (a->source.same_as(b->source)) &&
         (a->axis.same_as(b->axis)) && (a->condition.same_as(b->condition));
}

Schedule Schedule::normalize() {
  Schedule sn = copy();
  InfoUpdater::channelCount = 0;
  return sn;
}

}  // namespace TVM
