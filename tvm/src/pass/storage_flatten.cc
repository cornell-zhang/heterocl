/*!
 *  Copyright (c) 2016 by Contributors
 * \file storage_flatten.cc
 */
// Flattens storage from multi-dimensional array to 1D
// buffer access as in Halide pipeline.
#include <tvm/buffer.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_operator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target_info.h>
#include <unordered_map>
#include "../arithmetic/compute_expr.h"
#include "../runtime/thread_storage_scope.h"
#include "./arg_binder.h"
#include "./ir_util.h"

namespace TVM {
namespace ir {

using Halide::Internal::Region;
using intrinsic::tvm_address_of;
using runtime::StorageScope;
using runtime::ThreadScope;

Type String2Type(std::string& s) {
  if (s.front() == '\"' && s.back() == '\"') {
    s.erase(0, 1);
    s.pop_back();
  }
  std::istringstream is(s);
  halideir_type_code_t code = Type::Int;
  if (s.substr(0, 3) == "int") {
    code = Type::Int;
    s = s.substr(3);
  } else if (s.substr(0, 4) == "uint") {
    code = Type::UInt;
    s = s.substr(4);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float;
    s = s.substr(5);
  } else if (s == "handle") {
    return Handle();
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  int bits = 32, lanes = 1;
  if (sscanf(s.c_str(), "%dx%d", &bits, &lanes) == 0) {
    LOG(FATAL) << "unknown type " << s;
  }
  return Type(code, bits, lanes);
}

class StorageFlattener : public IRMutator {
 public:
  explicit StorageFlattener(Map<Tensor, Buffer> extern_buffer,
                            int cache_line_size) {
    for (auto kv : extern_buffer) {
      BufferEntry e;
      e.buffer = kv.second;
      e.external = true;
      buf_map_[TensorKey{kv.first->op, kv.first->value_index}] = e;
    }
    cache_line_size_ = cache_line_size;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<Variable>());
      VarExpr buf_var(it->second.node_);
      if (has_stencil_) outputs_.insert(buf_var);
      return Store::make(buf_var, op->value, op->index, op->predicate);
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::double_buffer_scope) {
      Operation func(op->node.node_);
      Stmt body = Mutate(op->body);
      for (int i = 0; i < func->num_outputs(); ++i) {
        TensorKey key{func, i};
        auto it = buf_map_.find(key);
        CHECK(it != buf_map_.end())
            << "Cannot find allocated buffer for " << key.f;
        body = AttrStmt::make(it->second.buffer->data, op->attr_key, op->value,
                              body);
      }
      return body;
    } else if (op->attr_key == attr::thread_extent) {
      IterVar iv(op->node.node_);
      ThreadScope ts = ThreadScope::make(iv->thread_tag);
      curr_thread_scope_.push_back(ts);
      Stmt stmt = IRMutator::Mutate_(op, s);
      curr_thread_scope_.pop_back();
      return stmt;
    } else if (op->attr_key == attr::buffer_bind_scope) {
      return HandleBufferBindScope(op);
    } else if (op->attr_key == attr::buffer_dim_align) {
      Tensor tensor(op->node.node_);
      const Call* tuple = op->value.as<Call>();
      CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
      TensorKey key{tensor->op, tensor->value_index};
      auto& vinfo = dim_align_[key];
      int dim = tuple->args[0].as<IntImm>()->value;
      if (static_cast<size_t>(dim) >= vinfo.size()) {
        vinfo.resize(dim + 1);
      }
      vinfo[dim].align_factor = tuple->args[1].as<IntImm>()->value;
      vinfo[dim].align_offset = tuple->args[2].as<IntImm>()->value;
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::opengl_stage_scope) {
      is_opengl_ = true;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Provide>();
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key.f;
    const BufferEntry& e = it->second;
    CHECK(!e.released) << "Read a buffer that is already out of scope";
    if (is_opengl_) {
      return Evaluate::make(Call::make(Type(), Call::glsl_texture_store,
                                       {e.buffer->data, op->value},
                                       Call::Intrinsic));
    } else {
      return e.buffer.vstore(e.RelIndex(op->args), op->value);
    }
  }

  Stmt Mutate_(const Realize* op, const Stmt& s) final {
    TensorKey key{op->func, op->value_index};
    if (buf_map_.count(key)) {
      // CHECK(buf_map_.at(key).external)
      //     << key.f << " not in external buffer bindings";
      return this->Mutate(op->body);
    } else {
      // create a buffer entry
      BufferEntry e;
      e.bounds = op->bounds;
      Array<Expr> shape;
      for (auto r : e.bounds) {
        shape.push_back(r->extent);
      }
      // deduce current storage scope.
      auto it = storage_scope_.find(op->func.get());
      CHECK(it != storage_scope_.end())
          << "Cannot find storage scope of " << op->func
          << " value_index=" << op->value_index;
      StorageScope skey;
      const std::string& strkey = it->second;
      if (strkey.length() == 0) {
        if (curr_thread_scope_.size() != 0) {
          skey.rank = curr_thread_scope_.back().rank + 1;
        }
      } else {
        skey = StorageScope::make(strkey);
      }

      // use small alignment for small arrays
      int32_t const_size =
          Allocate::constant_allocation_size(shape, key.GetName());
      int align = GetTempAllocaAlignment(op->type, const_size);
      if (skey.tag.length() != 0) {
        MemoryInfo info = GetMemoryInfo(skey.to_string());
        if (info.defined()) {
          align = (info->max_simd_bits + op->type.bits() - 1) / op->type.bits();
          CHECK_LE(const_size * op->type.bits(), info->max_num_bits)
              << "Allocation exceed bound of memory tag " << skey.to_string();
        }
      }
      Array<Expr> strides;
      if (dim_align_.count(key) != 0 && shape.size() != 0) {
        std::vector<Expr> rstrides;
        const std::vector<DimAlignInfo>& avec = dim_align_[key];
        int first_dim = 0;
        Expr stride = make_const(shape[first_dim].type(), 1);
        for (size_t i = shape.size(); i != 0; --i) {
          size_t dim = i - 1;
          if (dim < avec.size() && avec[dim].align_factor != 0) {
            Expr factor = make_const(stride.type(), avec[dim].align_factor);
            Expr offset = make_const(stride.type(), avec[dim].align_offset);
            stride = stride + (factor + offset - stride % factor) % factor;
            stride = ir::Simplify(stride);
          }
          rstrides.push_back(stride);
          stride = arith::ComputeExpr<Mul>(stride, shape[dim]);
        }
        strides = Array<Expr>(rstrides.rbegin(), rstrides.rend());
      }

      e.buffer = BufferNode::make(Var(key.GetName(), Handle()), op->type, shape,
                                  strides, Expr(), key.GetName(),
                                  skey.to_string(), align, 0);

      buf_map_[key] = e;
      Stmt body = this->Mutate(op->body);
      buf_map_[key].released = true;
      Stmt ret;

      Type dtype = e.buffer->dtype;
      if (strides.size() != 0) {
        int first_dim = 0;
        ret = Allocate::make(
            e.buffer->data, dtype,
            {arith::ComputeExpr<Mul>(e.buffer->strides[first_dim],
                                     e.buffer->shape[first_dim])},
            make_const(Bool(e.buffer->dtype.lanes()), true), body,
            Array<Stmt>(), Expr(), std::string(), op->init_values,
            op->is_const);
      } else {
        shape = e.buffer->shape;
        if (shape.size() == 0) {
          shape.push_back(make_const(Int(32), 1));
        }
        ret = Allocate::make(e.buffer->data, dtype, shape,
                             make_const(Bool(e.buffer->dtype.lanes()), true),
                             body, Array<Stmt>(), Expr(), std::string(),
                             op->init_values, op->is_const);
      }
      ret = AttrStmt::make(e.buffer->data, attr::storage_scope,
                           StringImm::make(e.buffer->scope), ret);
      return ret;
    }
  }

  Stmt Mutate_(const KernelDef* op, const Stmt& s) final {
    for (size_t i = 0; i < op->arg_tensors.size(); i++)
      kernel_arg_tensors_.push_back(op->arg_tensors[i]);

    Stmt body = this->Mutate(op->body);
    return KernelDef::make(op->args, op->arg_shapes, op->arg_types,
                           op->arg_tensors, body, op->ret_void, op->ret_type,
                           op->name, op->attributes);
  }

  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<Variable>());
      VarExpr buf_var(it->second.node_);
      if (has_stencil_) inputs_.insert(buf_var);
      return Load::make(op->type, buf_var, op->index, op->predicate);
    } else {
      return expr;
    }
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

  Expr Mutate_(const Call* op, const Expr& olde) final {
    Expr expr = IRMutator::Mutate_(op, olde);
    op = expr.as<Call>();
    if (op != nullptr && op->call_type == Call::Halide) {
      TensorKey key{op->func, op->value_index};
      auto it = buf_map_.find(key);
      CHECK(it != buf_map_.end())
          << "Cannot find allocated buffer for " << key.f;
      const BufferEntry& e = it->second;
      CHECK(!e.released) << "Read a buffer that is already out of scope";
      return e.buffer.vload(e.RelIndex(op->args), e.buffer->dtype);
    } else {
      return expr;
    }
  }

  Stmt Mutate_(const Prefetch* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Prefetch>();
    CHECK(op != nullptr);
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key.f;
    const BufferEntry& e = it->second;

    CHECK(!e.released) << "Read a buffer that is already out of scope";
    CHECK_EQ(e.buffer->shape.size(), op->bounds.size())
        << "Prefetch dim should be the same as buffer dim";

    int block_size = 1, elem_cnt = cache_line_size_ / e.buffer->dtype.bytes(),
        shape = 0;

    int starts = op->bounds.size() - 1;
    while (starts > 0 && arith::GetConstInt(e.buffer->shape[starts], &shape) &&
           elem_cnt >= block_size * shape) {
      block_size *= shape;
      starts--;
    }
    Expr stride(elem_cnt / block_size);

    Array<Expr> args;
    std::vector<VarExpr> vars;

    for (int i = op->bounds.size() - 1; i > starts; --i) {
      args.push_back(op->bounds[i]->min);
    }
    auto& func_name = op->func->func_name();
    vars.push_back(VarExpr(
        "prefetch." + func_name + "." + std::to_string(starts), Int(32)));
    args.push_back(op->bounds[starts]->min + stride * vars.back());
    for (int i = starts - 1; i >= 0; --i) {
      vars.push_back(
          VarExpr("prefetch." + func_name + "." + std::to_string(i), Int(32)));
      args.push_back(vars.back() + op->bounds[i]->min);
    }
    for (int i = starts; i >= 0; --i) {
      if (i < starts) {
        stmt = For::make(vars[i], 0, op->bounds[i]->extent, ForType::Serial,
                         DeviceAPI::Host, stmt);
      } else {
        Expr load = e.buffer.vload(e.RelIndex(args), e.buffer->dtype);
        Expr address =
            Call::make(Handle(), tvm_address_of, {load}, Call::PureIntrinsic);
        Expr prefetch = Call::make(op->type, Call::prefetch, {address, 0, 3, 1},
                                   Call::Intrinsic);
        stmt = Evaluate::make(prefetch);
        Expr extent = (op->bounds[i]->extent - 1) / stride + 1;
        stmt = For::make(vars[i], 0, extent, ForType::Serial, DeviceAPI::Host,
                         stmt);
      }
    }
    return stmt;
  }

  Stmt Mutate_(const Reuse* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Reuse>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<Variable>());
      VarExpr buf_var(it->second.node_);
      return Reuse::make(buf_var, op->body);
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const Partition* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Partition>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<Variable>());
      VarExpr buf_var(it->second.node_);
      return Partition::make(buf_var, op->dim, op->factor, op->partition_type);
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const Stencil* op, const Stmt& s) final {
    // check whether the stencil is updated
    if (op->inputs.size() == 0 && op->outputs.size() == 0) {
      if (has_stencil_) LOG(FATAL) << "Nested stencil is not supported";
      has_stencil_ = true;
      inputs_.clear();
      outputs_.clear();
      Stmt body = this->Mutate(op->body);
      has_stencil_ = false;
      Array<VarExpr> new_inputs;
      Array<VarExpr> new_outputs;
      // TODO(sean): this is inefficent
      for (auto input : inputs_) {
        if (!outputs_.count(input)) new_inputs.push_back(input);
      }
      for (auto output : outputs_) {
        if (!inputs_.count(output)) new_outputs.push_back(output);
      }
      return Stencil::make(new_inputs, new_outputs, body, op->burst_width,
                           op->unroll_factor, op->num_iteration);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  // The specific tensor data layout is not determined before
  // StorageFlatten pass. We use buffer_bind_scope
  // to specify before hand we want to bind a subregion
  // of tensor to a symbolic buffer, which get used in extern.
  //
  // Example:
  //
  // realize A in range [i*4, extent=10) {
  //   bind Ab to A in [i*4+1, extent=4) {
  //     call_func(Ab.ptr, Ab.shape[0])
  //   }
  // }
  //
  // After StorageFlatten
  //
  // alloc A[10]
  //   call(A + 1,  4)
  //
  // Buffer is a protocol to declare specific
  // data layout and shape we expect.
  // So this function need to check:
  // - If the bind range is within the realize range
  // - If we can match the requirement of buffer
  // - Remap variables such as Ab.ptr to the actual value.
  //
  // Here are a few possible failure cases:
  // - Buffer is declared to have constant shape,
  //   but we try to bind it to a different one.
  // - Buffer is declared to be compact(no strides)
  //   but this binded region is a subregion of
  //   a matrix(tensor), which means it requires strides.
  //
  // We do support a few relaxed case, such as bindingx
  // region with shape [1, 1, n, m] to buffer with shape [n, m]
  Stmt HandleBufferBindScope(const AttrStmt* op) {
    Array<NodeRef> arr(op->node.node_);
    CHECK_EQ(arr.size(), 2U);
    const BufferNode* buffer = arr[0].as<BufferNode>();
    const TensorNode* tensor = arr[1].as<TensorNode>();
    const Call* tuple = op->value.as<Call>();
    CHECK(buffer && tensor);
    CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
    // special handle for kernel args
    for (size_t i = 0; i < kernel_arg_tensors_.size(); i++) {
      if (tensor->op == kernel_arg_tensors_[i]) return this->Mutate(op->body);
    }
    TensorKey key{tensor->op, tensor->value_index};
    if (!buf_map_.count(key)) return this->Mutate(op->body);
    CHECK(buf_map_.count(key)) << "Cannot find buffer of " << tensor->op
                               << " value=" << tensor->value_index;
    const BufferEntry& be = buf_map_.at(key);
    // FIXME: reuse binding tensor
    // CHECK(!be.released);
    CHECK_EQ(tuple->args.size(), be.buffer->shape.size() * 2);
    Array<Expr> begins, extents;
    if (be.bounds.size() != 0) {
      CHECK_EQ(tuple->args.size(), be.bounds.size() * 2);
      for (size_t i = 0; i < be.buffer->shape.size(); ++i) {
        begins.push_back(
            arith::ComputeExpr<Sub>(tuple->args[2 * i], be.bounds[i]->min));
        extents.push_back(tuple->args[2 * i + 1]);
      }
    } else {
      for (size_t i = 0; i < tuple->args.size(); i += 2) {
        begins.push_back(tuple->args[i]);
        extents.push_back(tuple->args[i + 1]);
      }
    }
    Buffer slice = be.buffer.MakeSlice(begins, extents);
    if (buffer->strides.size() == 0) {
      CHECK_EQ(slice->strides.size(), 0U)
          << "Trying to bind compact buffer to strided one strides="
          << slice->strides;
    } else {
      slice = slice.MakeStrideView();
    }
    // start binding
    ArgBinder binder(&var_remap_);
    binder.BindBuffer(Buffer(arr[0].node_), slice, buffer->name, true);
    // Apply the remaps
    // TODO(sean): fix this?
    // Stmt body = MergeNest(binder.asserts(), op->body);
    Stmt body = MergeNest(binder.init_nest(), op->body);
    body = this->Mutate(body);
    // remove the binds
    for (const Var& v : binder.defs()) {
      var_remap_.erase(v.get());
    }
    return body;
  }
  // The buffer entry in the flatten map
  struct DimAlignInfo {
    int align_factor{0};
    int align_offset{0};
  };
  // The buffer entry in the flatten map
  struct BufferEntry {
    // the buffer of storage
    Buffer buffer;
    // the bounds of realization, can be null, means everything
    Region bounds;
    // Whether the buffer is external
    bool external{false};
    // Whether we are out of allocation bounds and buffer get released.
    bool released{false};
    // relative index
    inline Array<Expr> RelIndex(Array<Expr> args) const {
      if (bounds.size() != 0) {
        Array<Expr> index;
        CHECK_EQ(bounds.size(), args.size());
        for (size_t i = 0; i < bounds.size(); ++i) {
          index.push_back(args[i] - bounds[i]->min);
        }
        return index;
      } else {
        return args;
      }
    }
  };
  // The buffer assignment map
  // Variable remap
  std::unordered_map<const Variable*, Expr> var_remap_;
  // Buffer map
  std::unordered_map<TensorKey, BufferEntry> buf_map_;
  // Dimension alignment
  std::unordered_map<TensorKey, std::vector<DimAlignInfo> > dim_align_;
  // Storage scope
  std::unordered_map<const Node*, std::string> storage_scope_;
  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;
  // The size of cacheline
  int cache_line_size_;
  // The current stage is an OpenGL shader.
  bool is_opengl_{false};
  // for stencil analysis
  bool has_stencil_{false};
  std::unordered_set<VarExpr, ExprHash, ExprEqual> inputs_;
  std::unordered_set<VarExpr, ExprHash, ExprEqual> outputs_;
  // for KernelDef
  std::vector<FunctionRef> kernel_arg_tensors_;
};

Stmt StorageFlatten(Stmt stmt, Map<Tensor, Buffer> extern_buffer,
                    int cache_line_size) {
  stmt = StorageFlattener(extern_buffer, cache_line_size).Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
