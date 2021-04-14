/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_lang.cc
 */
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule.h>
#include <unordered_set>
#include "../op/op_util.h"
#include "./compute_primitive.h"
#include "./graph.h"

namespace TVM {

namespace {

using namespace ir;
class IterVarBodyUpdater final : public IRMutator {
 public:
  IterVarBodyUpdater(const IterVar& var) : var_(var) {}
  Stmt Mutate(Stmt stmt) final {
    if (const For* op = stmt.as<For>()) {
      if (op->loop_var.get() == var_->var.get()) {
        Stmt body = AttrStmt::make(op->loop_var, "dataflow",
                                   StringImm::make("null"), op->body);
        return For::make(var_->var, op->min, op->extent, op->for_type,
                         op->device_api, body, op->annotate_keys,
                         op->annotate_values);
      } else {
        return IRMutator::Mutate(stmt);
      }
    }
    return IRMutator::Mutate(stmt);
  }

 private:
  const IterVar& var_;
};

class AttachedStagesUpdater final : public IRVisitor {
 public:
  AttachedStagesUpdater(const Map<NodeRef, Stage>& stage_buf_map, Stage& stage)
      : stage_buf_map_(stage_buf_map), stage_(stage) {}

  void Visit_(const AttrStmt* op) {
    if (op->attr_key == attr::attach_scope) {
      if (stage_buf_map_.count(op->node)) {
        stage_->attached_stages.push_back(stage_buf_map_.at(op->node));
      }
    }
    IRVisitor::Visit_(op);
  }

 private:
  const Map<NodeRef, Stage>& stage_buf_map_;
  Stage& stage_;
};

// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Node* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

size_t FindLeafVar(ArrayNode* all_vars, ArrayNode* leaf_vars,
                   const IterVar& v) {
  size_t pos = FindNodeRef(leaf_vars, v);
  if (pos < leaf_vars->data.size()) return pos;

  if (FindNodeRef(all_vars, v) < all_vars->data.size()) {
    LOG(FATAL) << "Operate on iter var " << v
               << "that has already been splitted";
  } else {
    LOG(FATAL) << "Operate on iter var " << v
               << "that is not part of the schedule";
  }
  return 0;
}

size_t FindIterVarPos(const Array<IterVar>& axis, const IterVar& v) {
  for (size_t i = 0; i < axis.size(); i++)
    if (axis[i]->var.get() == v->var.get()) return i;
  LOG(FATAL) << "IterVar " << v << " does not exist in the axis list";
  return -1;
}

void SubstituteStageStmts(std::vector<Stage> stages,
                          std::unordered_map<const Variable*, Expr>& sub) {
  for (size_t i = 0; i < stages.size(); i++) {
    SubstituteStageStmts(stages[i]->attached_stages, sub);
    auto node = stages[i]->op.as<ExternOpNode>();
    Stmt body = op::Substitute(node->body, sub);
    stages[i]->op = ExternOpNode::make(node->name, node->tag, node->axis,
                                       node->inputs, node->input_placeholders,
                                       node->output_placeholders, body);
  }
}

void Split(StageNode* self, IterVar parent, Expr factor, Expr nparts,
           IterVar* p_outer, IterVar* p_inner) {
  // Check if split is valid
  CHECK(parent->iter_type == kDataPar || parent->iter_type == kCommReduce ||
        parent->iter_type == kOrdered)
      << "Cannot split on " << IterVarType2String(parent->iter_type);
  // outer loop after split
  Expr outer_min = Simplify(parent->dom->min / factor);
  Expr outer_extent = Simplify((parent->dom->extent + factor - 1) / factor);
  IterVar outer = IterVarNode::make(Range(outer_min, outer_extent),
                                    parent->var.copy_with_suffix(".outer"),
                                    parent->iter_type);
  // inner loop after split
  Expr inner_min = make_const(parent->var.type(), 0);
  Expr inner_extent = factor;
  IterVar inner = IterVarNode::make(Range(inner_min, inner_extent),
                                    parent->var.copy_with_suffix(".inner"),
                                    parent->iter_type);
  *p_outer = outer;
  *p_inner = inner;
  // mutate axis
  auto old_op = self->op.as<ExternOpNode>();
  Array<IterVar> old_axis = old_op->axis;
  Array<IterVar> new_axis;
  for (size_t i = 0; i < old_axis.size(); ++i) {
    if (old_axis[i].get() == parent.get()) {
      new_axis.push_back(outer);
      new_axis.push_back(inner);
    } else {
      new_axis.push_back(old_axis[i]);
    }
  }
  // mutate stmt
  std::unordered_map<const Variable*, Expr> sub;
  Stmt old_stmt = old_op->body;
  Stmt new_stmt =
      SplitLoop(old_stmt, parent, factor, nparts, outer, inner, sub);
  // construct a new op
  self->op = ExternOpNode::make(old_op->name, old_op->tag, new_axis,
                                old_op->inputs, old_op->input_placeholders,
                                old_op->output_placeholders, new_stmt);
  SubstituteStageStmts(self->attached_stages, sub);
}

void Fuse(StageNode* self, IterVar outer, IterVar inner, IterVar* p_target) {
  CHECK(outer->iter_type == kDataPar || outer->iter_type == kCommReduce ||
        outer->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(outer->iter_type);
  CHECK(inner->iter_type == kDataPar || inner->iter_type == kCommReduce ||
        inner->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(inner->iter_type);

  IterVarType iter_type = outer->iter_type;
  if (inner->iter_type > iter_type) iter_type = inner->iter_type;
  std::string fused_name =
      outer->var->name_hint + "." + inner->var->name_hint + ".fused";

  Expr min = Simplify(inner->dom->min + outer->dom->min * inner->dom->extent);
  Expr extent = Simplify(inner->dom->extent * outer->dom->extent);

  IterVar fused = IterVarNode::make(
      Range(min, extent), Var(fused_name, outer->var.type()), iter_type);

  *p_target = fused;
  auto old_op = self->op.as<ExternOpNode>();
  Array<IterVar> old_axis = old_op->axis;
  Array<IterVar> new_axis;
  for (size_t i = 0; i < old_axis.size(); ++i) {
    if (old_axis[i].get() == outer.get()) {
      new_axis.push_back(fused);
    } else if (old_axis[i].get() == inner.get()) {
      continue;
    } else {
      new_axis.push_back(old_axis[i]);
    }
  }
  // mutate stmt
  std::unordered_map<const Variable*, Expr> sub;
  Stmt old_stmt = old_op->body;
  Stmt new_stmt = FuseLoop(old_stmt, outer, inner, fused, sub);
  // construct a new op
  self->op = ExternOpNode::make(old_op->name, old_op->tag, new_axis,
                                old_op->inputs, old_op->input_placeholders,
                                old_op->output_placeholders, new_stmt);
  // update all statements
  SubstituteStageStmts(self->attached_stages, sub);
}

void Reorder(StageNode* self, const Array<IterVar>& order) {
  std::unordered_set<IterVar> seen_var;
  for (IterVar iv : order) {
    CHECK(iv->iter_type == kDataPar || iv->iter_type == kCommReduce ||
          iv->iter_type == kThreadIndex)
        << "Cannot reorder IterVar(" << IterVarType2String(iv->iter_type)
        << ")";
    CHECK_EQ(seen_var.count(iv), 0)
        << "Same axis can not appear more than once " << iv;
    seen_var.insert(iv);
  }
  auto old_op = self->op.as<ExternOpNode>();
  Array<IterVar> old_axis = old_op->axis;
  Array<IterVar> new_axis;
  std::vector<size_t> old_pos;
  for (size_t i = 0; i < order.size(); i++)
    old_pos.push_back(FindIterVarPos(old_axis, order[i]));
  std::vector<size_t> new_pos(old_pos);
  std::sort(new_pos.begin(), new_pos.end());
  size_t counter = 0;
  for (size_t i = 0; i < old_axis.size(); i++) {
    if (i == new_pos[counter]) {
      new_axis.push_back(order[counter]);
      counter++;
    } else {
      new_axis.push_back(old_axis[i]);
    }
  }
  Stmt old_stmt = old_op->body;
  Stmt new_stmt = ReorderLoop(old_stmt, order);
  self->op = ExternOpNode::make(old_op->name, old_op->tag, new_axis,
                                old_op->inputs, old_op->input_placeholders,
                                old_op->output_placeholders, new_stmt);
}

void ComputeAt(StageNode* producer, StageNode* consumer, const IterVar& var,
               size_t& attach_level) {
  auto producer_op = producer->op.as<ExternOpNode>();
  auto consumer_op = consumer->op.as<ExternOpNode>();
  Stmt producer_stmt = producer_op->body;
  Stmt consumer_stmt = consumer_op->body;
  Buffer producer_buf = producer_op->output_placeholders[0];
  std::unordered_map<const Variable*, Expr> sub;
  Stmt new_stmt = PerformComputeAt(producer_stmt, consumer_stmt, producer_buf,
                                   var, attach_level, sub);
  producer->op =
      ExternOpNode::make(producer_op->name, producer_op->tag, producer_op->axis,
                         producer_op->inputs, producer_op->input_placeholders,
                         producer_op->output_placeholders, producer_stmt);
  consumer->op =
      ExternOpNode::make(consumer_op->name, consumer_op->tag, consumer_op->axis,
                         consumer_op->inputs, consumer_op->input_placeholders,
                         consumer_op->output_placeholders, new_stmt);
  // update all statements
  SubstituteStageStmts(producer->attached_stages, sub);
}

void CreateStencil(StageNode* stage, int burst_width, int unroll_factor,
                   int num_iteration) {
  const ExternOpNode* op = stage->op.as<ExternOpNode>();
  std::unordered_set<VarExpr, ExprHash, ExprEqual> input_set;
  std::unordered_set<VarExpr, ExprHash, ExprEqual> output_set;
  Array<VarExpr> inputs;
  Array<VarExpr> outputs;
  Stmt body = Stencil::make(inputs, outputs, op->body, burst_width,
                            unroll_factor, num_iteration);
  stage->op =
      ExternOpNode::make(op->name, op->tag, op->axis, op->inputs,
                         op->input_placeholders, op->output_placeholders, body);
}

void CreateDataflow(StageNode* stage, IterVar var) {
  const ExternOpNode* op = stage->op.as<ExternOpNode>();
  Stmt body;
  if (var.defined()) {
    IterVarBodyUpdater itbu(var);
    body = itbu.Mutate(op->body);
  } else {
    body = AttrStmt::make(VarExpr("null"), "dataflow", StringImm::make("null"),
                          op->body);
  }
  stage->op =
      ExternOpNode::make(op->name, op->tag, op->axis, op->inputs,
                         op->input_placeholders, op->output_placeholders, body);
}

}  // namespace

Stage::Stage(Operation op) {
  auto n = std::make_shared<StageNode>();
  n->op = op;
  n->origin_op = op;
  n->all_iter_vars = op->root_iter_vars();
  // remove opaque var from leaf.
  Array<IterVar> clean;
  for (IterVar iv : n->all_iter_vars) {
    if (iv->iter_type != kOpaque) clean.push_back(iv);
    n->iter_var_exprs_before_reorder.push_back(iv->var);
    n->iter_var_exprs_after_reorder.push_back(iv->var);
  }
  if (clean.size() == n->all_iter_vars.size()) {
    n->leaf_iter_vars = n->all_iter_vars;
  } else {
    n->leaf_iter_vars = clean;
  }
  node_ = n;
}

bool Stage::is_scheduled() const {
  const StageNode* n = operator->();
  return !(n->relations.empty() && n->attach_type == kGroupRoot &&
           n->all_iter_vars.same_as(n->leaf_iter_vars));
}

Stage Stage::GetAttachSpec() const {
  Stage attach_spec = *this;
  while (attach_spec->attach_type == kGroupRoot &&
         attach_spec->group.defined()) {
    attach_spec = attach_spec->group;
  }
  return attach_spec;
}

Stage& Stage::set_scope(std::string scope) {  // NOLINT(*)
  (*this)->scope = scope;
  return *this;
}

Stage& Stage::compute_at(Stage parent, IterVar scope) {  // NOLINT(*)
  (*this)->attach_type = kScope;
  (*this)->attach_ivar = scope;
  (*this)->attach_stage = parent;
  parent->attached_stages.push_back(*this);
  size_t attach_level = 0;
  ComputeAt(operator->(), parent.operator->(), scope, attach_level);
  (*this)->attach_level = attach_level - 1;
  return *this;
}

Stage& Stage::compute_inline() {  // NOLINT(*)
  (*this)->attach_type = kInline;
  return *this;
}

Stage& Stage::compute_root() {  // NOLINT(*)
  (*this)->attach_type = kGroupRoot;
  return *this;
}

Stage& Stage::bind(IterVar ivar, IterVar thread_ivar) {  // NOLINT(*)
  StageNode* self = operator->();
  CHECK(ivar->iter_type == kDataPar || ivar->iter_type == kCommReduce)
      << "Cannot bind " << IterVarType2String(ivar->iter_type) << " to thread";
  CHECK(thread_ivar->iter_type == kThreadIndex)
      << "Cannot rebase by " << IterVarType2String(ivar->iter_type)
      << ", only thread axis is allowed so far";
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, ivar);

  auto it = self->iter_var_attrs.find(ivar);
  std::shared_ptr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = std::make_shared<IterVarAttrNode>(*(*it).second.operator->());
    if (n->bind_thread.defined() && !n->bind_thread.same_as(thread_ivar)) {
      LOG(WARNING) << "Axis " << ivar << " is already bind to another thread "
                   << n->bind_thread;
    }
  } else {
    n = std::make_shared<IterVarAttrNode>();
  }
  n->bind_thread = thread_ivar;
  self->iter_var_attrs.Set(ivar, IterVarAttr(n));
  return *this;
}

Stage& Stage::env_threads(Array<IterVar> threads) {
  StageNode* self = operator->();
  /* TODO(Sean): remove the whole function
  CHECK(self->op.defined() && self->op.as<ScanOpNode>())
      << "env_threads is only valid for composite ops such as ScanOp";
  */
  CHECK_EQ(self->env_threads.size(), 0U) << "Already set env_threads";
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  std::vector<std::shared_ptr<Node> > temp;
  for (IterVar iv : threads) {
    temp.push_back(iv.node_);
  }
  leaf_vars->data.insert(leaf_vars->data.begin(), temp.begin(), temp.end());
  all_vars->data.insert(all_vars->data.end(), temp.begin(), temp.end());
  self->env_threads = threads;
  return *this;
}

Stage& Stage::set_store_predicate(Expr predicate) {
  StageNode* self = operator->();
  self->store_predicate = predicate;
  return *this;
}

Stage& Stage::split(IterVar parent, Expr factor, IterVar* p_outer,
                    IterVar* p_inner) {
  Split(operator->(), parent, factor, Expr(), p_outer, p_inner);
  return *this;
}

Stage& Stage::split_by_nparts(IterVar parent, Expr nparts, IterVar* p_outer,
                              IterVar* p_inner) {
  Split(operator->(), parent, Expr(), nparts, p_outer, p_inner);
  return *this;
}

Stage& Stage::fuse(IterVar outer, IterVar inner, IterVar* p_target) {
  Fuse(operator->(), outer, inner, p_target);
  return *this;
}

Stage& Stage::reorder(const Array<IterVar>& order) {
  Reorder(operator->(), order);
  return *this;
}

Stage& Stage::tile(IterVar x_parent, IterVar y_parent, Expr x_factor,
                   Expr y_factor, IterVar* p_x_outer, IterVar* p_y_outer,
                   IterVar* p_x_inner, IterVar* p_y_inner) {
  split(x_parent, x_factor, p_x_outer, p_x_inner);
  split(y_parent, y_factor, p_y_outer, p_y_inner);
  reorder(Array<IterVar>({*p_x_outer, *p_y_outer, *p_x_inner, *p_y_inner}));
  return *this;
}

inline void SetIterVarAttr(StageNode* self, IterVar var,
                           IterVarAttrNode* node) {
  auto old_op = self->op.as<ExternOpNode>();
  Stmt old_stmt = old_op->body;
  Stmt new_stmt = UpdateIterVarAttr(old_stmt, var, node);
  self->op = ExternOpNode::make(old_op->name, old_op->tag, old_op->axis,
                                old_op->inputs, old_op->input_placeholders,
                                old_op->output_placeholders, new_stmt);
}

Stage& Stage::vectorize(IterVar var) {
  std::shared_ptr<IterVarAttrNode> node = std::make_shared<IterVarAttrNode>();
  node->iter_type = kVectorized;
  SetIterVarAttr(operator->(), var, node.get());
  return *this;
}

Stage& Stage::unroll(IterVar var) {
  std::shared_ptr<IterVarAttrNode> node = std::make_shared<IterVarAttrNode>();
  node->iter_type = kUnrolled;
  SetIterVarAttr(operator->(), var, node.get());
  return *this;
}

Stage& Stage::unroll(IterVar var, const Expr& factor) {
  std::shared_ptr<IterVarAttrNode> node = std::make_shared<IterVarAttrNode>();
  node->iter_type = kUnrolled;
  node->for_loop_annotate_keys.push_back(ir::StringImm::make("factor"));
  node->for_loop_annotate_values.push_back(factor);
  SetIterVarAttr(operator->(), var, node.get());
  return *this;
}

Stage& Stage::parallel(IterVar var) {
  std::shared_ptr<IterVarAttrNode> node = std::make_shared<IterVarAttrNode>();
  node->iter_type = kParallelized;
  SetIterVarAttr(operator->(), var, node.get());
  return *this;
}

Stage& Stage::pipeline(IterVar var, const Expr& initiation_interval_value) {
  std::shared_ptr<IterVarAttrNode> node = std::make_shared<IterVarAttrNode>();
  node->iter_type = kPipelined;
  node->for_loop_annotate_keys.push_back(
      ir::StringImm::make("initiation_interval"));
  node->for_loop_annotate_values.push_back(initiation_interval_value);
  SetIterVarAttr(operator->(), var, node.get());
  return *this;
}

Stage& Stage::split_annotate(IterVar var, Expr factor) {
  std::shared_ptr<IterVarAttrNode> node = std::make_shared<IterVarAttrNode>();
  node->for_loop_annotate_keys.push_back(ir::StringImm::make("split_factor"));
  node->for_loop_annotate_values.push_back(factor);
  SetIterVarAttr(operator->(), var, node.get());
  return *this;
}

Stage& Stage::split_by_nparts_annotate(IterVar var, Expr nparts) {
  std::shared_ptr<IterVarAttrNode> node = std::make_shared<IterVarAttrNode>();
  node->for_loop_annotate_keys.push_back(ir::StringImm::make("split_nparts"));
  node->for_loop_annotate_values.push_back(nparts);
  SetIterVarAttr(operator->(), var, node.get());
  return *this;
}

Stage& Stage::stencil(int burst_width, int unroll_factor, int num_iteration) {
  CreateStencil(operator->(), burst_width, unroll_factor, num_iteration);
  return *this;
}

Stage& Stage::dataflow(IterVar var) {
  CreateDataflow(operator->(), var);
  return *this;
}

Stage& Stage::pragma(IterVar var, const std::string& pragma_type) {
  if (pragma_type == "unroll") {
    this->unroll(var);
  } else if (pragma_type == "vectorize") {
    this->vectorize(var);
  }
  return *this;
}

Stage& Stage::prefetch(const Tensor& tensor, IterVar var, Expr offset) {
  StageNode* self = operator->();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, var);
  auto it = self->iter_var_attrs.find(var);
  std::shared_ptr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = std::make_shared<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = std::make_shared<IterVarAttrNode>();
  }
  n->prefetch_data.push_back(tensor);
  n->prefetch_offset.push_back(offset);
  self->iter_var_attrs.Set(var, IterVarAttr(n));
  return *this;
}

Stage& Stage::storage_align(IterVar axis, int factor, int offset) {
  return *this;
}

Stage& Stage::double_buffer() {
  StageNode* self = operator->();
  CHECK(!self->is_output) << "Cannot apply double buffer on output";
  self->double_buffer = true;
  return *this;
}

Stage& Stage::opengl() {
  CHECK(!is_scheduled()) << "Must be a fresh schedule";
  StageNode* self = operator->();

  auto all_iter_vars = self->all_iter_vars;  // curr version of all_iter_vars
  CHECK(!all_iter_vars.empty()) << "At least one iter var";

  // Fuse all data parallel dimensions to 1.
  IterVar fused = all_iter_vars[0];
  for (size_t i = 1; i != all_iter_vars.size(); ++i) {
    auto iter_var = all_iter_vars[i];
    switch (iter_var->iter_type) {
      case IterVarType::kDataPar: {
        fuse(fused, all_iter_vars[i], &fused);
        break;
      }
      case IterVarType::kThreadIndex: {
        LOG(ERROR) << "A fresh schedule shouldn't have thread index iter var";
        break;
      }
      case IterVarType::kCommReduce:
      case IterVarType::kOrdered:
      case IterVarType::kOpaque: {
        break;
      }
      default: {
        LOG(ERROR) << "Invalid iter var type "
                   << IterVarType2String(iter_var->iter_type);
        break;
      }
    }
  }

  // Bind the only dimension to threadIdx.x.
  bind(fused, thread_axis(Range(nullptr), "threadIdx.x"));

  // Mark this stage as OpenGL.
  (*this)->is_opengl = true;

  return *this;
}

Stage CopyStage(const Stage& s) {
  std::shared_ptr<StageNode> n = std::make_shared<StageNode>(*s.operator->());
  return Stage(n);
}

Schedule Schedule::copy() const {
  // map of stages.
  const ScheduleNode* self = operator->();
  std::unordered_map<Stage, Stage, NodeHash, NodeEqual> smap;
  std::shared_ptr<ScheduleNode> n = std::make_shared<ScheduleNode>();
  n->outputs = self->outputs;
  // Copy the stages.
  for (Stage s : self->stages) {
    Stage scopy = CopyStage(s);
    smap[s] = scopy;
    n->stages.push_back(scopy);
  }
  for (Stage g : self->groups) {
    Stage gcopy = CopyStage(g);
    smap[g] = gcopy;
    n->groups.push_back(gcopy);
  }
  // Remaps the reference relations.
  for (auto kv : self->stage_map) {
    n->stage_map.Set(kv.first, smap.at(kv.second));
  }
  for (Stage s : n->stages) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end())
          << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  for (Stage s : n->groups) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end())
          << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  return Schedule(n);
}

Stage Schedule::operator[](const Operation& op) {
  auto it = (*this)->stage_map.find(op);
  CHECK(it != (*this)->stage_map.end())
      << "Cannot find Stage for operator " << op << " in the schedule";
  return (*it).second;
}

Stage LeastCommonAncestor(Stage g1, Stage g2) {
  if (!g1.defined()) return g1;
  if (!g2.defined()) return g2;
  if (g1.same_as(g2)) return g1;
  Stage g = g1;
  while (g.defined()) {
    if (g.same_as(g2)) return g2;
    g = g->group;
  }
  g = g2;
  while (g.defined()) {
    if (g.same_as(g1)) return g1;
    g = g->group;
  }
  return g;
}

Array<Tensor> RemapTensor(ScheduleNode* self, const Array<Tensor>& arr) {
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  Array<Tensor> ret;
  for (Tensor t : arr) {
    if (!op2stage_cache.count(t->op.get())) {
      CHECK(self->stage_map.count(t->op))
          << "Given tensor is not in the schedule plan";
      t = self->stage_map[t->op]->op.output(t->value_index);
    }
    ret.push_back(t);
  }
  return ret;
}

// Group the schedule stages.
Stage Schedule::create_group(const Array<Tensor>& outputs,
                             const Array<Tensor>& inputs, bool include_inputs) {
  ScheduleNode* self = operator->();
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  // Get the ops.
  Array<Operation> ops = schedule::GetSubGraph(
      RemapTensor(self, outputs), RemapTensor(self, inputs), include_inputs);
  // local counter entry
  // Automatically initialize to 0 during creation.
  struct Entry {
    int count{0};
  };
  // Map of group->touched counter
  std::unordered_map<Stage, Entry, NodeHash, NodeEqual> counter;
  // The parent group;
  Stage parent_group;
  // Detect common parent and child.
  for (size_t i = 0; i < ops.size(); ++i) {
    Operation op = ops[i];
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage op_group = it->second->group;
    if (i == 0) {
      parent_group = op_group;
    } else {
      parent_group = LeastCommonAncestor(parent_group, op_group);
    }
    if (op_group.defined()) {
      ++counter[op_group].count;
    }
  }
  // Create the new group stage.
  Stage gstage(std::make_shared<StageNode>());
  gstage->group = parent_group;
  if (parent_group.defined()) {
    ++parent_group->num_child_stages;
  }
  // Propagate the counter statistics from by checking if subgroup
  // Is full and propagate.
  std::vector<Stage> stack;
  for (auto& kv : counter) {
    if (!kv.first.same_as(parent_group)) {
      if (kv.first->num_child_stages == kv.second.count) {
        stack.push_back(kv.first);
      }
    }
  }
  while (!stack.empty()) {
    Stage g = stack.back();
    stack.pop_back();
    if (g->group.defined() && !g->group.same_as(parent_group)) {
      Entry& e = counter[g->group];
      ++e.count;
      if (e.count == g->group->num_child_stages) {
        stack.push_back(g->group);
      }
    }
  }
  // Verification and remappig the subgroups.
  for (auto& kv : counter) {
    if (kv.first.same_as(parent_group)) continue;
    CHECK_EQ(kv.first->num_child_stages, kv.second.count)
        << "Trying to group region "
        << "that intersect with an already existed group";
    if (kv.first->group.same_as(parent_group)) {
      Stage s = kv.first;
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Remap the group of op stages.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage s = it->second;
    if (s->group.same_as(parent_group)) {
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Correct the attach to keep everything in group.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage s = it->second;
    if (s->attach_type == kScope) {
      Stage cg = LeastCommonAncestor(s->attach_stage->group, gstage);
      if (!cg.same_as(gstage)) {
        LOG(WARNING) << "group invalidates some previous compute_at relation "
                     << " and keeps things to be computed inside the group";
        s.compute_root();
      }
    }
  }

  self->groups.push_back(gstage);
  return gstage;
}

void ScheduleNode::InvalidateCache() { op2stage_cache_.clear(); }

void ScheduleNode::InitCache() {
  if (op2stage_cache_.size() == stages.size()) return;
  InvalidateCache();
  for (Stage s : stages) {
    if (s->op.defined()) {
      op2stage_cache_[s->op.get()] = s;
    }
  }
  CHECK_EQ(op2stage_cache_.size(), stages.size());
}

Schedule ScheduleNode::make(Array<Operation> ops) {
  auto n = std::make_shared<ScheduleNode>();
  Schedule sch(n);
  n->outputs = ops;
  auto g = schedule::CreateReadGraph(n->outputs, sch);
  Array<Operation> post_order = schedule::PostDFSOrder(n->outputs, g);
  // output set.
  std::unordered_set<Operation> output_set;
  for (Operation x : ops) {
    output_set.insert(x);
  }
  for (Operation op : post_order) {
    Stage stage(op);
    stage->is_output = output_set.count(op) != 0;
    n->stages.push_back(stage);
    n->stage_map.Set(op, stage);
    if (const ExternOpNode* node = op.as<ExternOpNode>())
      n->stage_buff_map.Set(node->output_placeholders[0], stage);
  }
  for (Stage stage : n->stages) {
    AttachedStagesUpdater visitor(n->stage_buff_map, stage);
    if (const ExternOpNode* node = stage->op.as<ExternOpNode>())
      visitor.Visit(node->body);
  }
  return sch;
}

IterVarRelation RebaseNode::make(IterVar parent, IterVar rebased) {
  auto n = std::make_shared<RebaseNode>();
  n->parent = parent;
  n->rebased = rebased;
  return IterVarRelation(n);
}

TVM_REGISTER_NODE_TYPE(StageNode);
TVM_REGISTER_NODE_TYPE(IterVarAttrNode);
TVM_REGISTER_NODE_TYPE(RebaseNode);
TVM_REGISTER_NODE_TYPE(ScheduleNode);

// Printer
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<StageNode>([](const StageNode* op, IRPrinter* p) {
      if (op->op.defined()) {
        p->stream << "stage(" << op->origin_op->name << ", " << op << ")";
      } else {
        p->stream << "group-stage(" << op << ")";
      }
    })
    .set_dispatch<IterVarAttrNode>([](const IterVarAttrNode* op, IRPrinter* p) {
      p->stream << IterVarType2String(op->iter_type);
    })
    .set_dispatch<RebaseNode>([](const RebaseNode* op, IRPrinter* p) {
      p->stream << "rebase(";
      p->stream << "parent=";
      p->print(op->parent);
      p->stream << ", rebased=";
      p->print(op->rebased);
      p->stream << ')';
    })
    .set_dispatch<ScheduleNode>([](const ScheduleNode* op, IRPrinter* p) {
      p->stream << "schedule(" << op << ")";
    });
}  // namespace TVM
