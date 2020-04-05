/*!
 *  Copyright (c) 2020 by Contributors
 * \file schedule_dataflow_rewrite.cc
 */
#include <tvm/buffer.h>
#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <regex>
#include <unordered_set>
#include "./message_passing.h"
#include "../pass/ir_util.h"
#include "../arithmetic/compute_expr.h"

namespace TVM {

using namespace ir;

// find first occurance location in leaf
template<typename T>
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
  explicit InStageMover(const Expr& scope,
                        const int index) :
      scope_{scope}, index_(index) {}

    Stmt Mutate_(const For* op, const Stmt& s) {
      if (counter == index_) {
        return AttrStmt::make(
            VarExpr(), attr::device_scope, scope_, s);
      } else {
        counter += 1;
        return For::make(
            op->loop_var, op->min, op->extent, op->for_type, op->device_api,
            this->Mutate(op->body), op->annotate_keys, op->annotate_values);
      }
    }

 private:
  const Expr& scope_;
  const int index_;
  int counter{0};
};

// The replacer of data load.
class LoadReplacer : public ir::IRMutator {
 public:
  explicit LoadReplacer(
      const std::unordered_map<const Variable*, Buffer>& vsub)
      : vsub_(vsub) {}

  Expr Mutate_(const Load* op, const Expr& e) {
    auto it = vsub_.find(op->buffer_var.get());
    if (it != vsub_.end())  
      return Load::make(op->type, VarExpr(it->second.node_),
                        op->index, op->predicate);
    return e;
  }

 private:
  const std::unordered_map<const Variable*, Buffer>& vsub_;
};

// The replacer of cache.
class VarReplacer : public ir::IRMutator {
 public:
  explicit VarReplacer(
      const std::unordered_map<const Variable*, Expr>& vsub)
      : vsub_(vsub) {}

  Expr Mutate_(const Variable* op, const Expr& e) {
    auto it = vsub_.find(op);
    if (it != vsub_.end()) return it->second;
    return e;
  }

 private:
  const std::unordered_map<const Variable*, Expr>& vsub_;
};

// update the kernel stmt annotation
class KernelMarker : public ir::IRMutator {
 public:
  explicit KernelMarker(Buffer buffer) :
      buf_(buffer) {}
  Stmt Mutate_(const KernelStmt* op, const Stmt& s) {
    // used in stream inference ir pass
    // to update the allocate stmt attr
    auto keys = op->annotate_keys;
    auto values = op->annotate_values;
    auto var = VarExpr(buf_->data.node_);
    for (int i = 0; i < (signed)op->args.size(); i++) {
      if (op->args[i].same_as(var)) {
        keys.push_back(StringImm::make("target_buffer_pos"));
        values.push_back(i);
      }
    }
    return KernelStmt::make(op->args, op->name, keys, values);
  }
  Buffer buf_;
};

// data serialization in sender 
class LoopBuilder : public ir::IRMutator {
 public:
  explicit LoopBuilder(
      Buffer load_buffer, Array<IterVar> old_axis,
      Expr& access_pattern, 
      const std::unordered_map<const Variable*, Expr>& range)
      : load_buffer_(load_buffer), old_axis_(old_axis),
        access_pattern_(access_pattern), range_(range) {}

  // mutate nested for loops
  Stmt Mutate_(const For* op, const Stmt& s) {
    std::vector<Stmt> nested_loop;
    Stmt next_s = s;
    std::unordered_set<const Variable*> loop_vars;
    while (const For* for_ = next_s.as<For>()) {
      nested_loop.push_back(next_s);
      next_s = for_->body;
      loop_vars.insert(for_->loop_var.get());
    }
    // replace load expr in stream stmt 
    Expr index = access_pattern_;
    auto target_load = index.as<Load>();
    Expr expr = IRMutator::Mutate_(target_load, index);
    // create new iter var array
    auto stream_op = next_s.as<StreamStmt>();
    auto old_load = stream_op->value.as<Load>();
    auto new_load = Load::make(old_load->type, old_load->buffer_var,
                               target_load->index, old_load->predicate);
    auto new_stmt = StreamStmt::make(stream_op->buffer_var, new_load, 
                                     stream_op->stream_type, 
                                     stream_op->depth, stream_op->annotate_keys,
                                     stream_op->annotate_values); 
    // replace itervar in target load expr
    int count = 0;
    std::string name = load_buffer_->name;
    for (auto it = range_.begin(); it != range_.end(); it++) {
      int extent = it->second.as<IntImm>()->value + 1;
      IterVar new_iv = IterVarNode::make(
          Range(0, extent), Var(name + std::to_string(count)), kDataPar);
      new_axis_.push_back(new_iv);
      vsub_[it->first] = new_iv->var;
      new_stmt = For::make(VarExpr(new_iv->var.node_), 0, extent,
                           ForType::Serial, DeviceAPI::None, new_stmt);
      count = count + 1;
    }
    return VarReplacer(vsub_).Mutate(new_stmt);
  }

  // record variables in expr
  Expr Mutate_(const Variable* op, const Expr& e) {
    auto it = range_.find(op);
    CHECK(it != range_.end()) 
      << "not found itervar ptr in range_";
    return e;
  }

  // new axis arr for extern op
  Array<IterVar> new_axis_;

 private:
  Buffer load_buffer_;
  Array<IterVar> old_axis_;
  Expr& access_pattern_;
  const std::unordered_map<const Variable*, Expr>& range_;
  std::unordered_map<const Variable*, Expr> vsub_;
};

class ParentStmtCollector final : public IRMutator {
  public:
    ParentStmtCollector(
        const VarExpr& target_buf,
        const VarExpr& reuse_buf,
        const std::string& parent_name,
        const IterVar& axis) 
      : target_buf_(target_buf), 
        reuse_buf_(reuse_buf), 
        parent_name_(parent_name),
        axis_(axis) { CHECK(target_buf.defined()); };

    Stmt Mutate_(const For* op, const Stmt& s) {
      if (op->loop_var.get() == axis_->var.get()) {
        const AttrStmt* attr = op->body.as<AttrStmt>();
        Stmt attr_stmt = AttrStmt::make(
            reuse_buf_,
            "attach_scope",
            StringImm::make(parent_name_),
            attr->body);
        attr_stmt = AttrStmt::make(
            attr->node,
            attr->attr_key,
            attr->value,
            attr_stmt);
        Stmt reuse_stmt = Reuse::make(target_buf_, attr_stmt);
        return For::make(
            op->loop_var, op->min, op->extent, op->for_type, op->device_api,
            reuse_stmt, op->annotate_keys, op->annotate_values);
      } else {
        return For::make(
            op->loop_var, op->min, op->extent, op->for_type, op->device_api,
            IRMutator::Mutate(op->body), op->annotate_keys, op->annotate_values);
      }
    }

  private:
    const VarExpr& target_buf_;
    const VarExpr& reuse_buf_;
    const std::string& parent_name_;
    const IterVar& axis_;
};

Expr InjectPredicate(const Array<Expr>& predicates,
                     Expr body) {
  using ir::Reduce;
  using ir::Select;
  if (predicates.size() == 0) return body;
  const Reduce* reduce = body.as<Reduce>();
  if (reduce) {
    std::shared_ptr<Reduce> n = std::make_shared<Reduce>(*reduce);
    n->condition = n->condition && arith::ComputeReduce<ir::And>(predicates, Expr());
    return Expr(n);
  }
  return Select::make(arith::ComputeReduce<ir::And>(predicates, Expr()),
                      body,
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

// update channel info of kernel def
class InfoUpdater final : public IRMutator {
  public: 
    static int channelCount;
    InfoUpdater(
        const int arg_pos,
        const int channel_depth,
        const int channel_index,
        const int is_sender) 
      : arg_pos_(arg_pos), 
        channel_depth_(channel_depth),
        channel_index_(channel_index),
        is_sender_(is_sender) { }

    Stmt Mutate_(const KernelDef* op, const Stmt& s) {
      Array<Expr> arr = op->channels;
      CHECK(op->channels.size() % 4 == 0)
        << "(pos, channel index, depth) pair number mismatch";
      arr.push_back(IntImm::make(Int(32), arg_pos_));
      arr.push_back(IntImm::make(Int(32), channel_index_));
      arr.push_back(IntImm::make(Int(32), channel_depth_));
      arr.push_back(IntImm::make(Int(32), is_sender_));
      return KernelDef::make(op->args, op->arg_shapes, 
                             op->arg_types, op->arg_tensors,
                             op->body, op->ret_void,
                             op->ret_type, op->name, arr);
    }
  private:
    const int arg_pos_;
    const int channel_depth_;
    int channel_index_{0}; 
    const int is_sender_; 
};

// Initialize static channel count
int InfoUpdater::channelCount = 0;

// stream buffer data to kernel stage 
void Schedule::to_stage(const Tensor& target,
                        /*kernel def stage*/ Stage dest,
                        /*position index*/int arg_pos,
                        StreamType stream_type,
                        int channel_depth,
                        std::string name) {
  Stage target_stage = (*this)[target];
  Buffer target_buffer;

  // target stage as kernel def operator (receiver) 
  if (auto op = target_stage->op.as<ExternOpNode>()) {
    target_buffer = op->output_placeholders[0];
    Buffer input_buffer = op->input_placeholders[0];

    // mutate to add new sender
    target_stage->op = ExternOpNode::make(op->name,
                                          "",
                                          Array<IterVar>(),
                                          op->inputs,
                                          op->input_placeholders,
                                          op->output_placeholders,
                                          op->body);
    // update dest stage body for data stream in 
    const ExternOpNode* destOp = dest->op.as<ExternOpNode>();
    dest->op = ExternOpNode::make(destOp->name, destOp->tag,
                                  destOp->axis, destOp->inputs,
                                  destOp->input_placeholders,
                                  Array<Buffer>(),
                                  destOp->body);
  }
}

// stream data between hardware modules  
void Schedule::stream_to(const Tensor& target,
                         Stage dest,
                         Stage source,
                         Array<Expr> stream_pos,
                         StreamType stream_type,
                         int channel_depth, 
                         std::string new_name) {
  Stage target_stage = (*this)[target];
  std::vector<Stage> consumers; 
  size_t num_stage = (*this)->stages.size();
  Buffer target_buffer;
  const ExternOpNode* destOp = dest->op.as<ExternOpNode>();
  const ExternOpNode* srcOp = source->op.as<ExternOpNode>();

  // update kernel def and scope 
  const PlaceholderOpNode* op = target_stage->op.as<PlaceholderOpNode>();
  bool is_placeholder = op ? true : false;
  if (is_placeholder) {
    for (size_t i = 0; i < num_stage; i++) {
      Stage s = (*this)->stages[i];
      if (const ExternOpNode* op = s->op.as<ExternOpNode>()) {
        for (size_t j = 0; j < op->inputs.size(); j++) {
          if (target == op->inputs[j]) {
            target_buffer = op->input_placeholders[j];
            consumers.push_back(s);
          }
        }
      }
    }
  } else { // mark device scope of consumers & update kernel stmts 
    const ExternOpNode* op = target_stage->op.as<ExternOpNode>();
    target_buffer = op->output_placeholders[0];
    consumers.push_back(target_stage);
    for (size_t i = 0; i < num_stage; i++) {
      Stage s = (*this)->stages[i];
      if (const ExternOpNode* op = s->op.as<ExternOpNode>()) {
        for (size_t j = 0; j < op->inputs.size(); j++) {
          if (target_buffer == op->input_placeholders[j]) {
            consumers.push_back(s); // mark buffer in calls
          }
        }
      }
    }
  }

  // inter-stage data movement 
  if (stream_pos.size() == 0) {

    if (destOp == srcOp) {
      // mutate loop body (attr_value indicates self-loop)
      VarExpr node(target_buffer->data.node_);
      Stmt dest_body = AttrStmt::make(
          node,
          attr::device_scope,
          IntImm::make(Int(32), 0),
          destOp->body);
      dest->op = ExternOpNode::make(destOp->name, destOp->tag,
                                    destOp->axis, destOp->inputs,
                                    destOp->input_placeholders,
                                    destOp->output_placeholders,
                                    dest_body);
    } else {
      // create common channel buffer
      VarExpr node(target_buffer->data.node_);
      InfoUpdater::channelCount += 1;
      auto ch_index = InfoUpdater::channelCount;

      Stmt dest_body = AttrStmt::make(
          node,
          attr::device_scope,
          IntImm::make(Int(32), ch_index),
          destOp->body);
      dest->op = ExternOpNode::make(destOp->name, destOp->tag,
                                    destOp->axis, destOp->inputs,
                                    destOp->input_placeholders,
                                    destOp->output_placeholders,
                                    dest_body);
      
      Stmt src_body = AttrStmt::make(
          node,
          attr::device_scope,
          IntImm::make(Int(32), -1 * ch_index),
          srcOp->body);
      source->op = ExternOpNode::make(srcOp->name, srcOp->tag,
                                      srcOp->axis, srcOp->inputs,
                                      srcOp->input_placeholders,
                                      srcOp->output_placeholders,
                                      src_body);
    }
    
  } else { // streaming between kernel defs
    CHECK(stream_pos.size() == 2) << "missing pos index";
    int destPos = stream_pos[0].as<IntImm>()->value;
    int srcPos  = stream_pos[1].as<IntImm>()->value;

    // create common channel buffer
    InfoUpdater::channelCount += 1;
    auto ch_index = InfoUpdater::channelCount;

    // update annotation in kernek def stmt 
    int dest_status = 0;
    int src_status = 1;
    // self-feedback mode
    if (destOp == srcOp) { 
      src_status = -1;
      dest_status = -1;
    }

    InfoUpdater destMutator(destPos, ch_index, 
                    channel_depth, dest_status);
    InfoUpdater srcMutator(srcPos, ch_index, 
                    channel_depth, src_status);

    Stmt dest_body = destMutator.Mutate(destOp->body);
    dest->op = ExternOpNode::make(destOp->name, destOp->tag,
                                  destOp->axis, destOp->inputs,
                                  destOp->input_placeholders,
                                  destOp->output_placeholders,
                                  dest_body);

    Stmt src_body = srcMutator.Mutate(srcOp->body);
    source->op = ExternOpNode::make(srcOp->name, srcOp->tag,
                                  srcOp->axis, srcOp->inputs,
                                  srcOp->input_placeholders,
                                  srcOp->output_placeholders,
                                  src_body);
  }

  // store info in kernel stmt
  for (auto s : consumers) {
    const ExternOpNode* op = s->op.as<ExternOpNode>();
    Stmt body = op->body;
    if (!is_placeholder) { 
      KernelMarker marker(target_buffer);
      body = marker.Mutate(body);
    }
    s->op = ExternOpNode::make(op->name,
                               op->tag,
                               op->axis,
                               op->inputs,
                               op->input_placeholders,
                               op->output_placeholders,
                               body);
  }
}

// move substages within HeteroCL stage
void Schedule::stage_move(
    Stage parent,
    DeviceType device_type,
    StreamType stream_type,
    int channel_depth, 
    int occur_index) {

  Expr scope;
  switch (device_type) {
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
  CHECK(scope.defined()) <<  "unsopport device ";
  const ExternOpNode* op = parent->op.as<ExternOpNode>();
  CHECK(op) << parent << " not a extern op";
  Stmt body = InStageMover(scope,
                  occur_index).Mutate(op->body);

  // result must be moved back before stage ends  
  parent->op = ExternOpNode::make(
      op->name,
      op->tag,
      op->axis,
      op->inputs,
      op->input_placeholders,
      op->output_placeholders,
      body);
}

// annotate the tensor to be joined  
void Schedule::join_to(const Tensor& target,
                       Stage source,
                       Stage dest,
                       StreamType stream_type,
                       int channel_depth) {

  Stage target_stage = (*this)[target];
  size_t num_stage = (*this)->stages.size();
  Buffer target_buffer;

  const PlaceholderOpNode* op = target_stage->op.as<PlaceholderOpNode>();
  bool is_placeholder = op ? true : false;
  if (is_placeholder) {
    for (size_t i = 0; i < num_stage; i++) {
      Stage s = (*this)->stages[i];
      if (const ExternOpNode* op = s->op.as<ExternOpNode>()) {
        for (size_t j = 0; j < op->inputs.size(); j++) {
          if (target == op->inputs[j]) {
            target_buffer = op->input_placeholders[j];
          }
        }
      }
    }
  } else { // mark device scope of consumers & update kernel stmts 
    const ExternOpNode* op = target_stage->op.as<ExternOpNode>();
    target_buffer = op->output_placeholders[0];
  }

  CHECK(source.defined());
  const ExternOpNode* src_op = source->op.as<ExternOpNode>();
  CHECK(src_op) << "cannot join placeholder stage " << source;

  InfoUpdater::channelCount += 1;
  auto index = InfoUpdater::channelCount;

  CHECK(target_buffer.defined());
  VarExpr node(target_buffer->data.node_);

  if (dest.defined()) {
    // insert attr into collector op
    const ExternOpNode* dest_op = dest->op.as<ExternOpNode>();
    CHECK(dest_op) << "cannot join to placeholder stage " << dest;
    Stmt body = dest_op->body;

    Stmt dest_body = AttrStmt::make(
        node,
        attr::device_scope,
        IntImm::make(Int(32), index),
        dest_op->body);
    dest->op = ExternOpNode::make(dest_op->name, dest_op->tag,
                                  dest_op->axis, dest_op->inputs,
                                  dest_op->input_placeholders,
                                  dest_op->output_placeholders,
                                  dest_body);

  } else { // create result collector stage

  }
  Stmt src_body = AttrStmt::make(
      node,
      attr::device_scope,
      IntImm::make(Int(32), -1 * index),
      src_op->body);
  source->op = ExternOpNode::make(
          src_op->name, src_op->tag, src_op->axis, src_op->inputs,
          src_op->input_placeholders, src_op->output_placeholders, src_body);
}

// move data to device
Tensor Schedule::move_to(const Tensor& target,
                         Stage parent,
                         DeviceType device_type,
                         StreamType stream_type,
                         int channel_depth, 
                         int occurrence) {
  Stage target_stage = (*this)[target];
  std::vector<Stage> consumers; 
  size_t num_stage = (*this)->stages.size();
  size_t min_pos = num_stage;
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  Buffer target_buffer;

  // create producer and consumer stages for placeholder
  const PlaceholderOpNode* op = target_stage->op.as<PlaceholderOpNode>();
  bool is_placeholder = op ? true : false;

  if (is_placeholder) {
    min_pos = 0;
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
  } else { // move data generated by extern op 
    min_pos = FindNodeRef(stages, target_stage) + 1;
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

  if (parent.defined()) { // stream modified tensor 
    target_stage = parent; 
    min_pos = FindNodeRef(stages, parent) + 1;
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

  // create sender that writes into streaming channel 
  Array<Tensor> consumer_inputs;
  Array<Buffer> consumer_input_placeholders;
  Array<Buffer> consumer_output_placeholders;
  std::string consumer_name = target->op->name + ".channel";
  if (parent.defined()) consumer_name = target->op->name + ".update.channel";

  Buffer channel_buffer = BufferNode::make(
      Var(consumer_name, Handle()),
      target->dtype,
      target->shape,
      Array<Expr>(),
      Expr(),
      consumer_name,
      "", 0, 0);

  if (!parent.defined()) {
    consumer_inputs.push_back(target);
    consumer_input_placeholders.push_back(target_buffer);
  } else { 
    const ExternOpNode* prt = parent->op.as<ExternOpNode>();
    CHECK(prt) << "stage " << parent << " not extern op";
    consumer_inputs.push_back(parent->op.output(0));
    consumer_input_placeholders.push_back(prt->output_placeholders[0]);
  }
  consumer_output_placeholders.push_back(channel_buffer);

  // create statement index
  std::vector<Expr> csm_indices;
  std::vector<VarExpr> csm_loop_vars;
  for (size_t i = 0; i < target->shape.size(); i++) {
    VarExpr iter(target->op->name + std::to_string(i));
    csm_indices.push_back(iter);
    csm_loop_vars.push_back(iter);
  }

  Expr csm_index = FlattenIndices(csm_indices, target->shape); 
  Expr load_expr = Load::make(
      target->dtype,
      VarExpr(target_buffer.node_), 
      csm_index, 
      UIntImm::make(UInt(1), 1));

  Stmt consumer_body = StreamStmt::make(
      VarExpr(channel_buffer.node_),
      load_expr, stream_type, channel_depth);

  Array<IterVar> consumer_axis;
  for (size_t j = 0; j < target->shape.size(); j++) {
    auto iter = csm_loop_vars[j];
    consumer_axis.push_back(IterVarNode::make(
        Range(0, target->shape[j]), Var(iter.node_), kDataPar));
    consumer_body = For::make(
      VarExpr(iter.node_),
      0, target->shape[j],
      ForType::Serial,
      DeviceAPI::None,
      consumer_body);
  }

  // create new stage and return stream tensors 
  Operation consumer_op = ExternOpNode::make(
      consumer_name, 
      "",
      consumer_axis,
      consumer_inputs,
      consumer_input_placeholders,
      consumer_output_placeholders,
      consumer_body);
  Stage consumer_stage = Stage(consumer_op);
  if (static_cast<DeviceType>(device_type) == DeviceType::devHost)
    consumer_stage->device_type = DeviceType::devFPGA; 

  stages->data.insert(stages->data.begin() + min_pos, consumer_stage.node_);
  (*this)->stage_map.Set(consumer_op, consumer_stage);

  // build producer (receiver) stage 
  Array<Tensor> producer_inputs;
  Array<Buffer> producer_input_placeholders;
  Array<Buffer> producer_output_placeholders;

  // new buffer copy of original data 
  std::string producer_name = target->op->name + ".new";
  if (parent.defined()) producer_name = target->op->name + ".update.new";
  Buffer output_buffer = BufferNode::make(
      Var(producer_name, Handle()),
      target->dtype,
      target->shape,
      Array<Expr>(),
      Expr(),
      producer_name,
      "", 0, 0);
  // producer writes into original target buffer
  producer_inputs.push_back(consumer_op.output(0));
  producer_input_placeholders.push_back(channel_buffer);
  producer_output_placeholders.push_back(output_buffer);

  // create for loops for tensor init
  std::vector<Expr> indices;
  std::vector<VarExpr> loop_vars;
  for (size_t i = 0; i < target->shape.size(); i++) {
    VarExpr iter(target->op->name + std::to_string(i));
    indices.push_back(iter);
    loop_vars.push_back(iter);
  }
  Expr index = FlattenIndices(indices, target->shape); 
  // streaming producer tensor reading from channel 
  Expr stream = StreamExpr::make(target->dtype,
                                 VarExpr(channel_buffer.node_),
                                 stream_type, channel_depth);
  // save data to new allocated data buffer
  Stmt for_stmt = Store::make(VarExpr(output_buffer.node_),
                              stream, index,
                              UIntImm::make(UInt(1), 1));
  Array<IterVar> producer_axis;
  for (size_t j = 0; j < target->shape.size(); j++) {
    auto iter = loop_vars[j];
    producer_axis.push_back(IterVarNode::make(
        Range(0, target->shape[j]), Var(iter.node_), kDataPar));
    for_stmt = For::make(
        VarExpr(iter.node_),
        0, target->shape[j],
        ForType::Serial,
        DeviceAPI::None,
        for_stmt);
  }

  Stmt body = for_stmt;
  // same buffer under different device scoep 
  Tensor producer = ExternOpNode::make(
      producer_name, 
      "",
      producer_axis,
      producer_inputs,
      producer_input_placeholders,
      producer_output_placeholders,
      body).output(0);

  Stage producer_stage = Stage(producer->op);
  producer_stage->device_type = static_cast<DeviceType>(device_type); 
  size_t pos = FindNodeRef(stages, consumer_stage);
  stages->data.insert(stages->data.begin() + pos, producer_stage.node_);
  (*this)->stage_map.Set(producer->op, producer_stage);

  // add producer as output stage if output moved to host
  if (target_stage->is_output && 
      static_cast<DeviceType>(device_type) == DeviceType::devHost) {
    (*this)->outputs.push_back(producer->op);
    target_stage->is_output = false;
    producer_stage->is_output = true;
  }

  // update consumer stages with new tensor and buffer
  std::unordered_map<Tensor, Tensor> vsub;
  std::unordered_map<const Variable*, Buffer> vsub2newvar;
  vsub[target] = producer; 
  vsub2newvar[target_buffer->data.as<Variable>()] = output_buffer;
  
  for (Stage s : consumers) {
    CHECK(s->op.as<ExternOpNode>());
    Operation repl_op = s->op->ReplaceInputs(s->op, vsub);

    // udpate stage not having orginal tensor input  
    auto op = repl_op.as<ExternOpNode>();
    Stmt repl_body = LoadReplacer(vsub2newvar).Mutate(op->body);
    
    Array<Tensor> new_inputs;
    Array<Buffer> new_input_placeholders;
    if (parent.defined()) {
      new_inputs.push_back(producer);
      new_input_placeholders.push_back(output_buffer);
    } else {
      new_inputs = op->inputs;
      new_input_placeholders = op->input_placeholders;
    } 
    
    s->op = ExternOpNode::make(
                op->name,
                op->tag,
                op->axis,
                new_inputs,
                new_input_placeholders,
                op->output_placeholders,
                repl_body);
    (*this)->stage_map.Set(s->op, s);
  }
  producer_stage->group = target_stage->group;
  if (producer_stage->group.defined()) {
    ++producer_stage->group->num_child_stages;
  }
  return producer;
}

Tensor Schedule::reuse_at(const Tensor& target,
                          Stage parent,
                          IterVar axis,
                          std::string reuse_name) {
  const ExternOpNode* op = parent->op.as<ExternOpNode>();
  Array<Tensor> reuse_inputs, new_inputs;
  Array<Buffer> reuse_input_placeholders, new_input_placeholders;
  Array<Buffer> reuse_output_placeholders;
  Stmt new_body;
  // the input is just the target
  Buffer target_buf;
  for(size_t i = 0; i < op->inputs.size(); i++) {
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
      LOG(FATAL) << "var " << target 
                 << "not found in input buffers";
    }
  } else { // reuse target buffer varexpr
    target_var = VarExpr(target_buf->data.node_); 
  }

  // create an output buffer
  Buffer reuse_output_buf = BufferNode::make(
      Var(reuse_name, Handle()),
      target->dtype,
      Array<Expr>(),
      Array<Expr>(),
      Expr(),
      reuse_name,
      "",
      0, 0);
  reuse_output_placeholders.push_back(reuse_output_buf);
  // traverse the parent body and collect the new information
  ParentStmtCollector mutator(target_var, 
                              VarExpr(reuse_output_buf.node_), 
                              op->name, axis);
  new_body = mutator.Mutate(op->body);
  // create reuse tensor
  Tensor reuse = ExternOpNode::make(reuse_name,
                                    "",
                                    Array<IterVar>(),
                                    reuse_inputs,
                                    reuse_input_placeholders,
                                    reuse_output_placeholders,
                                    Evaluate::make(0)).output(0);
  // update parent stage
  new_inputs = op->inputs;
  new_inputs.push_back(reuse);
  new_input_placeholders = op->input_placeholders;
  new_input_placeholders.push_back(reuse_output_buf);
  parent->op = ExternOpNode::make(op->name,
                                  op->tag,
                                  op->axis,
                                  new_inputs,
                                  new_input_placeholders,
                                  op->output_placeholders,
                                  new_body);
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
  Buffer partition_buffer = BufferNode::make(
      Var(partition_name, Handle()),
      Int(32),
      Array<Expr>(),
      Array<Expr>(),
      Expr(),
      partition_name,
      "", 0, 0);
  if (is_placeholder) {
    partition_inputs.push_back(target);
    partition_input_placeholders.push_back(target_buffer);
  }
  partition_output_placeholders.push_back(partition_buffer);
  Tensor partition_tensor = ExternOpNode::make(
      partition_name,
      "",
      Array<IterVar>(),
      partition_inputs,
      partition_input_placeholders,
      partition_output_placeholders,
      body).output(0);
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
      s->op = ExternOpNode::make(
          op->name,
          op->tag,
          op->axis,
          new_inputs,
          new_input_placeholders,
          op->output_placeholders,
          op->body);
    } else {
      Stmt new_body = AttrStmt::make(
          VarExpr(partition_buffer.node_),
          "attach_scope",
          StringImm::make(target_buffer->name),
          op->body);
      s->op = ExternOpNode::make(
          op->name,
          op->tag,
          op->axis,
          new_inputs,
          new_input_placeholders,
          op->output_placeholders,
          new_body);
    }
  }
  return partition_tensor;
}

// Do not support reshaping the placeholders for now
void Schedule::reshape(const Tensor& target, Array<Expr> new_shape) {
  Stage target_stage = (*this)[target];
  const ExternOpNode* op = target_stage->op.as<ExternOpNode>();
  Buffer target_buffer = op->output_placeholders[0];
  // TODO: check the #elem is the same for both shapes
  target_buffer->shape = new_shape;
}

Tensor Schedule::cache_read(const Tensor& tensor,
                            const std::string& scope,
                            const Array<Operation>& readers) {
  (*this)->InvalidateCache();
  // create identity mapping.
  std::ostringstream os;
  os << tensor->op->name;
  if (tensor->op->num_outputs() != 1) {
    os << ".v" << tensor->value_index;
  }
  os << "." << scope;

  std::unordered_map<Tensor, Tensor> vsub;
  Stage s = operator[](tensor->op);
  Tensor sugar_tensor = s->op.output(tensor->value_index);
  Tensor cache = compute(sugar_tensor->shape, [&sugar_tensor](const Array<Var>& i) {
      return sugar_tensor(Array<Expr>(i.begin(), i.end()));
    }, os.str());
  vsub[sugar_tensor] = cache;

  std::unordered_map<Tensor, Tensor> vmap;
  for (Operation op : readers) {
    Stage s = operator[](op);
    Operation repl_op = s->op->ReplaceInputs(s->op, vsub);
    CHECK(!repl_op.same_as(s->op))
        << "Cannot find " << tensor
        << " in the inputs of " << s->op;
    vmap[s->op.output(0)] = repl_op.output(0);
    s->op = repl_op;
  }
  ReplaceDataFlow((*this)->stages, &vmap);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  Stage op_stage = operator[](tensor->op);
  size_t pos = FindNodeRef(stages, op_stage);
  Stage cache_stage = Stage(cache->op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1,
                      cache_stage.node_);
  (*this)->stage_map.Set(cache->op, cache_stage);
  // Update group
  cache_stage->group = op_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache;
}


// Cache write and relayout the data according to loop pattern
Tensor CacheWriteWithReLayout(Schedule sch,
                              const Tensor& tensor,
                              const std::string& scope) {
  sch->InvalidateCache();
  Stage orig_stage = sch[tensor->op];
  const ComputeOpNode* compute = orig_stage->op.as<ComputeOpNode>();

  std::unordered_set<IterVar> red_axis;
  for (IterVar iv : compute->reduce_axis) {
    red_axis.insert(iv);
  }
  std::unordered_map<IterVar, Range> dom_map;
  Array<IterVar> new_axis;

  for (IterVar iv : compute->axis) {
    dom_map[iv] = iv->dom;
  }
  schedule::PassDownDomain(orig_stage, &dom_map, true);
  std::unordered_map<const Variable*, Expr> vsub;
  std::unordered_map<const Variable*, Expr> vsub2newvar;
  std::vector<Expr> predicates;
  {
    // The source->cache
    std::unordered_map<IterVar, Expr> value_map;
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      CHECK_EQ(iv->iter_type, kDataPar)
          << "Can only relayout with in data parallel dimensions";
      Range dom = dom_map.at(iv);
      IterVar new_iv = IterVarNode::make(
          dom, iv->var.copy_with_suffix(".c"), iv->iter_type);
      new_axis.push_back(new_iv);
      if (is_one(dom->min)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
        vsub2newvar[iv->var.get()] = new_iv->var;
      }
    }
    // skip reduction iteration.
    std::unordered_set<IterVar> skip_bound_check;
    for (IterVar iv : compute->reduce_axis) {
      skip_bound_check.insert(iv);
    }
    schedule::PassUpIndex(orig_stage, dom_map, &value_map, true);
    predicates = schedule::MakeBoundCheck(
        orig_stage, dom_map, value_map, true, skip_bound_check);
    // The root axis
    for (IterVar iv : compute->axis) {
      vsub[iv->var.get()] = value_map.at(iv);
    }
  }
  Expr body = VarReplacer(vsub).Mutate(compute->body[tensor->value_index]);
  body = InjectPredicate(predicates, body);
  body = VarReplacer(vsub2newvar).Mutate(body);
  // The reader args
  Array<Expr> args;
  {
    // cache->compute
    std::unordered_map<IterVar, Expr> value_map;
    for (IterVar iv : compute->axis) {
      value_map[iv] = iv->var;
    }
    schedule::PassDownIndex(orig_stage, dom_map, &value_map, true);
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      args.push_back(value_map.at(iv));
    }
  }
  Operation cache_op = ComputeOpNode::make(
      compute->name + "." + scope, compute->tag, new_axis, {body});
  Tensor cache_tensor = cache_op.output(0);
  Operation orig_new_op = ComputeOpNode::make(
      compute->name, compute->tag, compute->axis,
      {cache_tensor(args)});
  // The replace of the dataflow
  std::unordered_map<Tensor, Tensor> vmap;
  vmap[orig_stage->op.output(0)] = orig_new_op.output(0);
  ReplaceDataFlow(sch->stages, &vmap);
  // mutate orig stage
  orig_stage->op = orig_new_op;
  orig_stage->all_iter_vars = orig_stage->op->root_iter_vars();
  orig_stage->leaf_iter_vars = orig_stage->all_iter_vars;
  orig_stage->relations = Array<IterVarRelation>();
  // create schedule for new cached stage.
  ArrayNode* stages = sch->stages.CopyOnWrite();
  size_t pos = FindNodeRef(stages, orig_stage);
  Stage cache_stage = Stage(cache_op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos,
                      cache_stage.node_);
  sch->stage_map.Set(cache_op, cache_stage);
  // Update group
  cache_stage->group = orig_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache_tensor;
}

Tensor Schedule::cache_write(const Tensor& tensor,
                             const std::string& scope) {
  (*this)->InvalidateCache();
  Stage orig_stage = operator[](tensor->op);
  const ComputeOpNode* compute = tensor->op.as<ComputeOpNode>();
  CHECK(compute)
      << "cache write only take ComputeOp as writers";
  CHECK_EQ(compute->num_outputs(), 1)
      << "cache write only support single output ComputeOp";

  return CacheWriteWithReLayout(*this, tensor, scope);
}

void RebaseNonZeroMinLoop(Schedule& sch) {
  std::unordered_map<IterVar, IterVar> rebase_map;
  for (Stage s : sch->stages) {
    if (s->attach_type == kInlinedAlready) continue;

    auto root_iter_vars = s->op->root_iter_vars();
    ArrayNode* leaf_vars = s->leaf_iter_vars.CopyOnWrite();
    for (IterVar iv : root_iter_vars) {
      size_t idx = FindNodeRef(leaf_vars, iv);
      auto it  = s->iter_var_attrs.find(iv);
      // don;t need to rebase path that are binded.
      if (it != s->iter_var_attrs.end() &&
          (*it).second->bind_thread.defined()) {
        continue;
      }
      if (idx < leaf_vars->data.size()) {
        // insert rebase
        IterVar rebased = IterVarNode::make(
            Range(), iv->var.copy_with_suffix(""), iv->iter_type);
        s->relations.push_back(RebaseNode::make(iv, rebased));
        if (s->iter_var_attrs.count(iv)) {
          s->iter_var_attrs.Set(rebased, s->iter_var_attrs.at(iv));
        }
        leaf_vars->data[idx] = rebased.node_;
        rebase_map[iv] = rebased;
      }
    }
  }
  // remap the parent relation
  for (Stage s : sch->stages) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      sch->extern_itervar_map[rebase_map.at(s->attach_ivar)] = s->attach_ivar;
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
  for (Stage s : sch->groups) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      sch->extern_itervar_map[rebase_map.at(s->attach_ivar)] = s->attach_ivar;
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
}

inline bool ReduceEqual(const ir::Reduce* a, const ir::Reduce* b) {
  return (a->combiner.same_as(b->combiner)) &&
         (a->source.same_as(b->source)) &&
         (a->axis.same_as(b->axis)) &&
         (a->condition.same_as(b->condition));
}

void InjectInline(ScheduleNode* sch) {
  sch->InvalidateCache();

  std::vector<Array<Expr> > new_body(sch->stages.size());
  std::vector<bool> changed(sch->stages.size(), false);
  // inline all the ops
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage stage = sch->stages[i - 1];
    if (stage->attach_type == kInline) {
      stage->attach_type = kInlinedAlready;
      Array<Var> args;
      Expr body;
      {
        // setup args
        const ComputeOpNode* compute = stage->op.as<ComputeOpNode>();
        CHECK(compute)
            << "can only inline compute op";
        for (auto iv : compute->axis) {
          args.push_back(iv->var);
        }
        CHECK_EQ(compute->body.size(), 1U)
            << "can only inline compute op with 1 output";
        body = compute->body[0];
      }
      for (size_t j = i; j < sch->stages.size(); ++j) {
        Stage s = sch->stages[j];
        const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
        if (compute) {
          if (!new_body[j].size()) {
            new_body[j] = s->op.as<ComputeOpNode>()->body;
          }
          if (new_body[j][0]->is_type<ir::Reduce>()) {
            // specially handle reduction inline for multiplre reductions.
            const ir::Reduce* reduce = new_body[j][0].as<ir::Reduce>();
            for (size_t k = 1; k < new_body[j].size(); ++k) {
              const ir::Reduce* reduce_ = new_body[j][k].as<ir::Reduce>();
              CHECK(reduce_);
              CHECK(ReduceEqual(reduce_, reduce))
                  << "The Reduce inputs of ComputeOp should "
                  << "have the same attribute except value_index";
            }
            Expr new_value = ir::Inline(ir::Evaluate::make(new_body[j][0]),
                                        stage->op, args, body).as<ir::Evaluate>()->value;
            if (!new_value.same_as(new_body[j][0])) {
              changed[j] = true;
              const ir::Reduce* r = new_value.as<ir::Reduce>();
              CHECK_EQ(new_body[j].size(), r->source.size());
              CHECK(r != nullptr);
              for (size_t k = 0; k < new_body[j].size(); ++k) {
                std::shared_ptr<ir::Reduce> n = std::make_shared<ir::Reduce>(*r);
                n->value_index = static_cast<int>(k);
                n->type = r->source[k].type();
                new_body[j].Set(k, Expr(n));
              }
            }
          } else {
            for (size_t k = 0; k < new_body[j].size(); ++k) {
              Expr new_value = ir::Inline(ir::Evaluate::make(new_body[j][k]),
                                          stage->op, args, body).as<ir::Evaluate>()->value;
              if (!new_value.same_as(new_body[j][k])) {
                new_body[j].Set(k, new_value);
                changed[j] = true;
              }
            }
          }
        }
      }
    }
  }
  std::unordered_map<Tensor, Tensor> repl;
  // rewrite dataflow
  for (size_t i = 0; i < sch->stages.size(); ++i) {
    Stage s = sch->stages[i];
    if (s->attach_type == kInlinedAlready) continue;
    if (new_body[i].size()) {
      // Logics from ReplaceDataFlow
      const ComputeOpNode* compute = sch->stages[i]->op.as<ComputeOpNode>();
      CHECK(compute);
      Operation op = s->op;
      if (changed[i]) {
        op = ComputeOpNode::make(
            compute->name, compute->tag, compute->axis, new_body[i]);
      }
      op = op->ReplaceInputs(op, repl);
      if (!op.same_as(s->op)) {
        for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
          repl[s->op.output(idx)] = op.output(idx);
          s->op = op;
        }
      }
    } else {
      Operation op = s->op->ReplaceInputs(s->op, repl);
      if (!op.same_as(s->op)) {
        for (int j = 0; j < op->num_outputs(); ++j) {
          repl[s->op.output(j)] = op.output(j);
        }
        s->op = op;
      }
    }
  }
}

Schedule Schedule::normalize() {
  Schedule sn = copy();
  InjectInline(sn.operator->());
  InfoUpdater::channelCount = 0;
  //RebaseNonZeroMinLoop(sn);
  return sn;
}

// Handle reduction factor.
Array<Tensor> Schedule::rfactor(const Tensor& tensor,
                                const IterVar& axis,
                                int factor_axis) {
  (*this)->InvalidateCache();
  using ir::Reduce;
  CHECK_EQ(axis->iter_type, kCommReduce)
      << "Can only factor reduction axis";
  Stage reduce_stage = operator[](tensor->op);
  const ComputeOpNode* compute_op = reduce_stage->op.as<ComputeOpNode>();
  CHECK(compute_op) << "Can only factor ComputeOp";
  ArrayNode* leaf_vars = reduce_stage->leaf_iter_vars.CopyOnWrite();
  {
    size_t axis_pos = FindNodeRef(leaf_vars, axis);
    CHECK_NE(axis_pos, leaf_vars->data.size())
        << "Cannot find IterVar " << axis << " in leaf iter vars";
  }
  // Find touched reduction axis.
  std::unordered_map<IterVar, int> touch_map;
  touch_map[axis] = 1;
  schedule::PassUpBitMaskOr(reduce_stage, &touch_map, true);
  schedule::PassDownBitMaskOr(reduce_stage, &touch_map, true);
  // skip reduction iteration.
  std::unordered_set<IterVar> skip_bound_check;
  // Verify normal axis are not touched.
  for (IterVar iv : compute_op->axis) {
    CHECK(!touch_map.count(iv))
        << "Factor axis touches normal axis.";
    skip_bound_check.insert(iv);
  }
  // Get the replace index
  std::unordered_map<IterVar, Range> dom_map;
  std::unordered_map<IterVar, Expr> value_map;
  for (IterVar iv : compute_op->reduce_axis) {
    if (touch_map.count(iv)) {
      dom_map[iv] = iv->dom;
    } else {
      skip_bound_check.insert(iv);
    }
  }
  schedule::PassDownDomain(reduce_stage, &dom_map, true);
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv)) {
      Range dom = dom_map.at(iv);
      if (is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
      }
    }
  }
  schedule::PassUpIndex(reduce_stage, dom_map, &value_map, true);
  std::vector<Expr> predicates = schedule::MakeBoundCheck(
      reduce_stage, dom_map, value_map, true, skip_bound_check);

  // Get the factored op node.
  const int factor_axis_pos = \
      factor_axis >= 0 ? factor_axis : static_cast<int>(compute_op->axis.size() + 1) + factor_axis;
  CHECK_LE(factor_axis_pos, compute_op->axis.size());
  auto n = std::make_shared<ComputeOpNode>();
  n->name = compute_op->name + ".rf";
  {
    // axis relacement.
    auto iv_node = std::make_shared<IterVarNode>();
    iv_node->dom = dom_map.at(axis);
    CHECK(is_zero(iv_node->dom->min))
        << "Can only factor reduction domain starting from 0";
    iv_node->var = axis->var;
    iv_node->iter_type = kDataPar;

    const int size = compute_op->axis.size();
    for (int idx = 0; idx < size; ++idx) {
      if (factor_axis_pos == idx) {
        n->axis.push_back(IterVar(iv_node));
      }
      n->axis.push_back(compute_op->axis[idx]);
    }
    if (factor_axis_pos == size) {
      n->axis.push_back(IterVar(iv_node));
    }
  }
  // predicate generation, copy not touched axis.
  int idx = tensor->value_index;
  const Reduce* reduce = compute_op->body[idx].as<Reduce>();
  CHECK(reduce) << "Can only rfactor non-inline reductions";
  predicates.push_back(reduce->condition);
  Expr predicate = arith::ComputeReduce<ir::And>(predicates, Expr());

  std::unordered_map<const Variable*, Expr> vsub;

  for (IterVar iv : compute_op->reduce_axis) {
    if (!touch_map.count(iv)) {
      n->reduce_axis.push_back(iv);
    } else {
      CHECK(value_map.count(iv));
      Expr index = value_map.at(iv);
      vsub[iv->var.get()] = index;
    }
  }

  // Copy touched axis.
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv) && !iv.same_as(axis)) {
      CHECK_EQ(iv->iter_type, kCommReduce);
      auto ncpy = std::make_shared<IterVarNode>(*iv.operator->());
      ncpy->dom = dom_map.at(iv);
      n->reduce_axis.push_back(IterVar(ncpy));
    }
  }
  VarReplacer replacer(vsub);
  Array<Expr> new_source = ir::UpdateArray(reduce->source,
    [&replacer] (const Expr& e) { return replacer.Mutate(e); });
  std::vector<Expr> body;
  for (size_t idx = 0; idx < reduce->source.size(); ++idx) {
    body.emplace_back(Reduce::make(reduce->combiner,
                                   new_source,
                                   n->reduce_axis,
                                   predicate,
                                   idx));
  }
  n->body = Array<Expr>(body);
  // refresh relations, keep the un-touched relations.
  Array<IterVarRelation> rels;
  for (IterVarRelation rel : reduce_stage->relations) {
    bool touched = false;
    if (const SplitNode* r = rel.as<SplitNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      if (touch_map.count(r->fused)) touched = true;
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
    if (!touched) {
      rels.push_back(rel);
    }
  }
  // initialize the factored stage.
  Operation factor_op(n);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t stage_pos = FindNodeRef(stages, reduce_stage);
  Stage factor_stage = Stage(factor_op);
  factor_stage->relations = rels;
  CHECK_LT(stage_pos, stages->data.size());
  stages->data.insert(stages->data.begin() + stage_pos,
                      factor_stage.node_);
  (*this)->stage_map.Set(factor_op, factor_stage);
  factor_stage->group = reduce_stage->group;
  if (factor_stage->group.defined()) {
    ++factor_stage->group->num_child_stages;
  }
  // Replace the old reduction.
  IterVar repl_red_axis = reduce_axis(
      dom_map.at(axis), axis->var->name_hint + ".v");
  Array<Tensor> factor_tensors;
  Array<Tensor> old_tensors;
  int size = factor_op->num_outputs();
  for (int idx = 0; idx < size; ++idx) {
    factor_tensors.push_back(factor_op.output(idx));
    old_tensors.push_back(reduce_stage->op.output(idx));
  }
  Array<Tensor> repl_tensors = compute(old_tensors[0]->shape,
    [&](const Array<Var>& i) {
      Array<Expr> indices;
      const int idx_size = static_cast<int>(i.size());
      for (int idx = 0; idx < idx_size; ++idx) {
        if (factor_axis_pos == idx) {
          indices.push_back(repl_red_axis->var);
        }
        indices.push_back(i[idx]);
      }
      if (factor_axis_pos == idx_size) {
          indices.push_back(repl_red_axis->var);
      }
      Array<Expr> factor_exprs;
      for (int idx = 0; idx < size; ++idx) {
        factor_exprs.push_back(factor_tensors[idx](indices));
      }
      Array<Expr> reductions;
      Array<IterVar> axis = {repl_red_axis};
      Expr cond = const_true();
      for (int idx = 0; idx < size; ++idx) {
        reductions.push_back(Reduce::make(reduce->combiner,
          factor_exprs, axis, cond, idx));
      }
      return reductions;
    }, reduce_stage->op->name + ".repl");

  std::unordered_map<Tensor, Tensor> vmap;
  for (int idx = 0; idx < size; ++idx) {
    vmap[old_tensors[idx]] = repl_tensors[idx];
  }
  ReplaceDataFlow((*this)->stages, &vmap);
  // revamp the reduction stage.
  reduce_stage->op = repl_tensors[0]->op;
  reduce_stage->all_iter_vars = repl_tensors[0]->op->root_iter_vars();
  reduce_stage->leaf_iter_vars = reduce_stage->all_iter_vars;
  reduce_stage->relations = Array<IterVarRelation>();
  return factor_tensors;
}

}  // namespace TVM
