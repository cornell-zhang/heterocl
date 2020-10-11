/*!
 *  Copyright (c) 2020 by Contributors
 * \file stream_inference.cc
 * \brief mutate ir for scheduling streaming ops
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <unordered_map>
#include "./ir_util.h"
#include <ir/IREquality.h>
#include <arithmetic/Substitute.h>

namespace TVM {
namespace ir {

using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;

struct IoInfo {
    DeviceType    dev_type;
    StorageType   storage_type; 
    int           mem_port{-1};
    StreamType    stream_type;
    int           channel_depth{-1}; 
    int           burst_len{-1}; 
};

// The stream information of a buffer
struct StreamInfo {
    vector<int>    index_array;
    vector<int>    depth_array;
    int            max_consumers{0}; // for producer buffer
};
    
inline Expr Type2Expr(const Type& t) {
  if (t.code()  == Type::Handle) 
    return StringImm::make("handle");
  std::ostringstream os;
  os << t;
  return StringImm::make(os.str());
}

// Substitute the target buffer consumers with channel buffers
class NewChannelGathers final : public IRMutator {
 public: 
  NewChannelGathers(vector<int> _index_array,
    string _target_buffer_name,
    StreamInfo _target_buffer_stream_info,
    unordered_map<int, VarExpr>&  _channel_index_to_new_buffers) :
    index_array(_index_array), 
    target_buffer_name(_target_buffer_name),
    target_buffer_stream_info(_target_buffer_stream_info),
    channel_index_to_new_buffers(_channel_index_to_new_buffers) {}

  Stmt Mutate(Stmt stmt) final {
    if (!hit_target_channel_load) {
        Stmt ret = IRMutator::Mutate(stmt);

        // Add temp to save value before the statement
        if (hit_target_channel_load) {
            if (search_first_stmt_with_target == 0) {
                HCL_DEBUG_LEVEL(2) << "Insert streaming channel reader of "
                    << target_buffer_name << " before "
                    << "the first Stmt consumer: "<< ret;

                // Loading data from the channel 
                // TODO: support multiple index case
                auto index = index_array[0]; 
                auto target_load_op = target_load_expr.as<Load>();
                CHECK(target_load_op);
                CHECK(channel_index_to_new_buffers.count(index));
                VarExpr channel_buf(channel_index_to_new_buffers[index].node_); 
                Expr new_load = Load::make(target_load_op->type, 
                    channel_buf, target_load_op->index, target_load_op->predicate);

                Stmt s;
                if (insert_conditional_load) {
                    HCL_DEBUG_LEVEL(2) << "[debug] Hmm.. Insert cond store...";
                    CHECK(condition_node.defined());
                    auto select_op = condition_node.as<Select>();
                    CHECK(select_op); 

                    // Merge the two conditional stores
                    if (auto store_op = ret.as<Store>()) {
                        HCL_DEBUG_LEVEL(2) << "[debug] Merge to single IfThenElse...";

                        Stmt first = Store::make(new_var, new_load, 0, UIntImm::make(UInt(1), 1));
                        Stmt new_first_store = Store::make(store_op->buffer_var, select_op->true_value, store_op->index, store_op->predicate);
                        first = Block::make(first, new_first_store);

                        Stmt second = Store::make(new_var, 0, 0, UIntImm::make(UInt(1), 1));
                        Stmt new_second_store = Store::make(store_op->buffer_var, select_op->false_value, store_op->index, store_op->predicate);
                        second = Block::make(second, new_second_store);
                        ret = IfThenElse::make(select_op->condition, first, second);
                    }
                     
                } else {
                    s = Store::make(new_var, new_load, 0, UIntImm::make(UInt(1), 1));
                    ret = Block::make(s, ret);
                }
                ret = Allocate::make(new_var, target_load_op->type, {1}, 
                        make_const(Bool(target_load_op->type.lanes()), true), ret);
                ret = AttrStmt::make(new_var, attr::storage_scope, 
                        StringImm::make("global"), ret);
            }
            search_first_stmt_with_target++;
        }
        return ret;
    }
    return IRMutator::Mutate(stmt);
  }

  // Here we should check whether the accessing 
  // sequence is the same as the producer
  Expr Mutate_(const Load* op, const Expr& e) {
    auto name = op->buffer_var.get()->name_hint;
    if (name == target_buffer_name) {
        if (hit_target_channel_load) {
            CHECK(target_load_access_indices.size() > 0);
            Expr prev_index = target_load_access_indices[0];

            // Same buffer access with different index
            if (!equal(op->index, prev_index)) {
              LOG(FATAL) << "Invalid FIFO consumer \"" << name << "\". "
                << "It has more than buffer "
                << "access with different indices: " << name << "["
                << prev_index << "], " << name 
                << "[" << op->index << "]..."; 
              return e;

            // Same buffer access with same index
            } else {
                CHECK(new_var.defined());
                return Load::make(op->type, new_var, 0, op->predicate);
            }
        }
        hit_target_channel_load = true;
        CHECK(new_var.defined());
        target_load_expr = e;
        target_load_access_indices.push_back(op->index);

        if (inside_select_node) {
            HCL_DEBUG_LEVEL(2) << "[debug] Hmm.. Found load inside the select expr...";
            insert_conditional_load = true;
        }
        return Load::make(op->type, new_var, 0, op->predicate);
    }
    return IRMutator::Mutate_(op, e);
  }

  // Handle the load inside Select case 
  Expr Mutate_(const Select* op, const Expr& e) {
    inside_select_node = true;
    Expr expr = IRMutator::Mutate_(op, e);
    condition_node = e;
    inside_select_node = false;
    return expr;
  }

  Stmt SubstituteBufferLoads(Stmt s) {
    new_var = VarExpr(target_buffer_name + ".temp");
    Stmt new_body = Mutate(s);
    return new_body;
  }

  vector<int> index_array;
  string target_buffer_name;
  StreamInfo  target_buffer_stream_info;
  unordered_map<int, VarExpr>& channel_index_to_new_buffers;

  VarExpr new_var;
  Expr target_load_expr;
  Expr condition_node;
  bool hit_target_channel_load{false};
  bool inside_select_node{false};
  bool insert_conditional_load{false};
  int search_first_stmt_with_target{0};
  vector<Expr> target_load_access_indices;
};

// Create new channels 
class NewChannelCreators final : public IRMutator {
 public: 
  NewChannelCreators(vector<int> _index_array,
    string _target_buffer_name,
    StreamInfo _target_buffer_stream_info,
    unordered_map<int, VarExpr>&  _channel_index_to_new_buffers,
    unordered_map<string, Type> _dtype) :
    index_array(_index_array), 
    target_buffer_name(_target_buffer_name),
    target_buffer_stream_info(_target_buffer_stream_info),
    channel_index_to_new_buffers(_channel_index_to_new_buffers),
    dtype(_dtype) {}

  Stmt Mutate_(const Store* op, const Stmt& s) {
    auto name = op->buffer_var.get()->name_hint;

    // Use a temp value to store the value into a temp
    // There should only be a signle store for the target buffer
    if (name == target_buffer_name) {
        CHECK(!buffer_created) << "Failure: trying to stream a tensor that "
            << "has been written for multiple times...";
        HCL_DEBUG_LEVEL(2) << "Found target buffer store of " << name;

        buffer_created = true;
        VarExpr temp(name + ".temp");
        CHECK(dtype.count(target_buffer_name));
        auto type = dtype[target_buffer_name];
        Stmt stmt = Store::make(temp, op->value, 0, op->predicate);
        
        // Create buffers for vars in index array
        for (size_t k = 0; k < index_array.size(); k++) {
            auto index = -1 * index_array[k];
            auto new_name = name + ".pipe." + std::to_string(index);
            VarExpr new_channel_buffer(new_name);
            channel_index_to_new_buffers[index] = new_channel_buffer;
            HCL_DEBUG_LEVEL(2) << "Adding new buffer " << new_name
                << " for channel #" << index << "...";

            // Create store nodes to save the temp var
            Expr e = Load::make(type, temp, 0, op->predicate);
            Stmt s = Store::make(new_channel_buffer, e, op->index, op->predicate);
            stmt = Block::make(stmt, s);
        }

        // Write back to origina buffer if some
        // consumers still reads from it
        if (write_back) {
            Expr e = Load::make(type, temp, 0, op->predicate);
            Stmt s = Store::make(op->buffer_var, e, op->index, op->predicate);
            stmt = Block::make(stmt, s);
        } else {
            unused_buffers.push_back(op->buffer_var);
            HCL_DEBUG_LEVEL(2) << " -- Not writting back to original buffer... "
                << "Buffer " << op->buffer_var << " became unused...";
        }
            

        stmt = Allocate::make(temp, type, {1}, 
                make_const(Bool(type.lanes()), true), stmt);
        stmt = AttrStmt::make(temp, attr::storage_scope, 
                StringImm::make("global"), stmt);
        return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt CreateBuffers(Stmt stmt, Array<Expr> shape) {
    write_back = ((int)index_array.size() 
        == target_buffer_stream_info.max_consumers) ? false : true;
    Stmt s = Mutate(stmt);

    // Add buffer allocation nodes
    // at the beginning of the producer stage (stream_scope attr)
    for (auto index : index_array) {
        
        index *= -1;
        CHECK(channel_index_to_new_buffers.count(index)) << index;
        VarExpr buf(channel_index_to_new_buffers.at(index).node_); 
        auto channel_index = index; 

        int channel_depth = -1;
        auto index_array = target_buffer_stream_info.index_array;
        for (size_t k = 0; k < index_array.size(); k++) {
            if (index_array[k] == channel_index) {
              channel_depth = target_buffer_stream_info.depth_array[k];
            }
        }
        CHECK(channel_depth != -1);
        CHECK(dtype.count(target_buffer_name));
        Type type = dtype[target_buffer_name];

        Stmt attr = StreamStmt::make(
              buf, 0, IntImm::make(Int(32), channel_index), 0,
              StreamType::ATTR, channel_depth, Array<Expr>(), Array<Expr>()); 
        Array<Stmt> attrs = { attr };
        s = Allocate::make(buf, type, shape, 
                   make_const(Bool(type.lanes()), true), s, attrs, Expr(), string());
        s = AttrStmt::make(buf, attr::storage_scope, 
                StringImm::make("global"), s);
    }

    return RemoveNoOp(s);
  }

  vector<int> index_array;
  string target_buffer_name;
  StreamInfo  target_buffer_stream_info;
  unordered_map<int, VarExpr>& channel_index_to_new_buffers;
  unordered_map<string, Type> dtype;

  bool buffer_created{false};
  bool write_back;
  vector<VarExpr> unused_buffers;
  
};

// Mutate the Allocate Stmt and add StreamStmt into the attr 
class AllocateAttrDecorator final : public IRMutator {
 public: 
  AllocateAttrDecorator(
    unordered_map<string, vector<int> > _global_channel_trace,
    unordered_map<string, StreamInfo>  _inter_stage_channels,
    unordered_map<string, Type> _dtype,
    unordered_map<string, Array<Expr> > _shape)
    : global_channel_trace(_global_channel_trace),
      inter_stage_channels(_inter_stage_channels), dtype(_dtype), shape(_shape) {}

  // Add StreamStmt as attributes to stream_scoped Allocate
  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();

    string name = op->buffer_var.get()->name_hint;
    if (global_channel_trace.count(name)) {
      HCL_DEBUG_LEVEL(2) << "Found Streaming Channel " << name;
      auto params = global_channel_trace[name];
      int channel_index = params[0];
      int channel_depth = params[1];
      Stmt attr = StreamStmt::make(
            op->buffer_var, 0, IntImm::make(Int(32), channel_index), 0,
            StreamType::ATTR, channel_depth, Array<Expr>(), Array<Expr>()); 
      Array<Stmt> attrs = op->attrs;
      attrs.push_back(attr);
      return Allocate::make(op->buffer_var, op->type, op->extents,
                            op->condition, op->body, attrs,
                            op->new_expr, op->free_function);
    }
    return stmt;
  }

  // Mutate the stage body (with in the producer)
  // 1. Add new buffers (decorated with StreamStmt) as channels
  // 2. Add temp value to store the read value from prodoucer buffer
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    if (op->attr_key == attr::stream_attrs) {
        VarExpr var(op->node.node_);
        string buffer_name = var.get()->name_hint;
        Stmt body = op->body;

        CHECK(op->value.as<IntImm>());
        int index =  op->value.as<IntImm>()->value;
        HCL_DEBUG_LEVEL(2) << "Adding channel index " << index 
            << " for tensor " << buffer_name << " into the array...";

        // Map from channel name to channel index
        // E.g. Tensor B : channel index -1 (producer)
        unordered_map<string, vector<int> > index_map;
        index_map[buffer_name].push_back(index);

        while (auto attr = body.as<AttrStmt>()) {
            if (attr->attr_key != attr::stream_attrs) {
                body = attr->body;
                continue;
            }

            VarExpr attr_var(attr->node.node_);
            string attr_name = attr_var.get()->name_hint;
            CHECK(attr->value.as<IntImm>());
            int attr_index =  attr->value.as<IntImm>()->value;

            if (index_map[attr_name].size() > 0) {
                int last_index = index_map[attr_name].back();
                CHECK(attr_index * last_index > 0) 
                    << "Tensor " << attr_name << " cannot be read and written "
                    << "at the same time";
            }
            HCL_DEBUG_LEVEL(2) << "Adding channel index " << attr_index 
                << " for tensor " << attr_name << " into the array...";
            index_map[attr_name].push_back(attr_index);
            body = attr->body;
        }

        // Mutate the body statement for each entry in the index map
        for (auto& kv : index_map) {
            
            auto buf_name = kv.first;
            CHECK(inter_stage_channels.count(buf_name));
            auto info = inter_stage_channels[buf_name];
            // Producers nested attrs
            // 1. Create new buffers (attributed with StreamStmt)
            //    before pushing into the producer buffer
            vector<int> index_array = kv.second;
            if (index_array.back() < 0) {
                HCL_DEBUG_LEVEL(2) << " -- Creating channel buffers on the producer side...";
                NewChannelCreators ncc(index_array, buf_name, info, 
                    channel_index_to_new_buffers, dtype);
                CHECK(shape.count(buf_name));
                auto buf_shape = shape[buf_name];
                body = ncc.CreateBuffers(body, buf_shape);

                if (ncc.unused_buffers.size() > 0) {
                    for (auto& buf : ncc.unused_buffers) {
                        unused_write_buffers.push_back(buf);
                    }
                }

            // Consumers nested attrs
            // 1. Used the new buffers created by producers
            //    to substitute the origin buffer read by the consumer
            } else {
                HCL_DEBUG_LEVEL(2) << " -- Substituting channel buffers for tensor " 
                    << buf_name << " on the consumer side...";
                NewChannelGathers ncg(index_array, buf_name, info,
                    channel_index_to_new_buffers);
                body = ncg.SubstituteBufferLoads(body);
            }
        }
        HCL_DEBUG_LEVEL(2) << body;
        return body;
    }
    return IRMutator::Mutate_(op, s);
  }

  unordered_map<string, vector<int> > global_channel_trace;
  unordered_map<string, StreamInfo>  inter_stage_channels;
  unordered_map<int, VarExpr>        channel_index_to_new_buffers;
  unordered_map<string, Type> dtype;
  unordered_map<string, Array<Expr> > shape;
  vector<VarExpr> unused_write_buffers;
};


// 1. Substitute old buffer with new buffers (i.e. the buffers
//    defined as kernel function arguments)
// 2. If the tensor is moved to host from device, remove the on-chip
//    buffer allocation and use a function arg buffer to replace it
class SubstituteBuffers final : public IRMutator {
 public: 
  SubstituteBuffers(unordered_map<const Variable*, VarExpr>& _vmap,
    unordered_map<string, VarExpr>& _remove)
  : vmap(_vmap), remove(_remove) {}

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();

    string name = op->buffer_var.get()->name_hint;
    if (remove.count(name)) {
      HCL_DEBUG_LEVEL(2) << "Lifting buffer (alloc) " << name;
      lifted_buffers.push_back(op->buffer_var);
      return op->body;
    }
    return stmt;
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    if (vmap.count(op->buffer_var.get())) {
        HCL_DEBUG_LEVEL(2) << "Substituting buffer (load) " << op->buffer_var;
        VarExpr new_var(vmap[op->buffer_var.get()].node_);
        return Load::make(op->type, new_var, op->index, op->predicate);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    Expr value = this->Mutate(op->value);
    string name = op->buffer_var.get()->name_hint;
    if (remove.count(name))  {
      HCL_DEBUG_LEVEL(2) << "Substituting buffer (store) " << name;
      VarExpr new_var(remove[name].node_);
      return Store::make(new_var, value, op->index, op->predicate);
    }
    if (vmap.count(op->buffer_var.get())) {
        HCL_DEBUG_LEVEL(2) << "Substituting buffer (store) " << op->buffer_var;
        VarExpr new_var(vmap[op->buffer_var.get()].node_);
        return Store::make(new_var, value, op->index, op->predicate);
    }
    return Store::make(op->buffer_var, value, op->index, op->predicate);
  }

  
  Stmt Mutate_(const Partition* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Partition>();
    if (vmap.count(op->buffer_var.get())) {
      HCL_DEBUG_LEVEL(2) << "Substituting buffer (partition) " << op->buffer_var;
      VarExpr new_var(vmap[op->buffer_var.get()].node_);
      return Partition::make(new_var, op->dim, op->factor, op->partition_type);
    } else {
      return stmt;
    }
  }

  unordered_map<const Variable*, VarExpr>& vmap;
  unordered_map<string, VarExpr>& remove;
  vector<VarExpr> lifted_buffers;

};

// Removed unused buffers 
// E.g. If a buffer (that is written and later read) is replaced 
// completely with streaming channel. Namely, all its consumers read
// from it through streaming channels, then the value will only go through 
// all these channels, and will not be written back to the original buffers.
// Thus the original buffer becomes unused...
class UnusedBufferRemover final : public IRMutator {
 public: 
  UnusedBufferRemover(vector<VarExpr>& _unused_vars) 
    : unused_vars(_unused_vars) {}

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    string target_name = op->buffer_var.get()->name_hint;
    for (auto& v : unused_vars) {
        if (target_name == v.get()->name_hint) {
          HCL_DEBUG_LEVEL(2) << "Removed unused var " << target_name;
          if (remove_producer) return this->Mutate(op->body);
        }
    }
    return stmt;
  }

  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<ProducerConsumer>();
    if (op->is_producer) {
        auto name = op->func->func_name();
        for (auto& v : unused_vars) {
            if (name == v.get()->name_hint) {
              HCL_DEBUG_LEVEL(2) << "[ debug ] Removed the producer stage of " 
                << name << ":\n" << op->body;
              if (remove_producer) return Evaluate::make(0);
            }
        }
    }
    return stmt;
  }

  vector<VarExpr>& unused_vars;
  bool remove_producer{false};
};

// 1. Create the KernelDef Stmt for device function by 
//    allocating the arg for IO args. For those ExternOpNode
//    output moved to host, we need to deallocate the buffers 
// 2. Create KernelStmt (i.e. dev function call)
class KernelDefCreator final : public IRMutator {
 public: 
  KernelDefCreator(unordered_map<string, IoInfo>& _dev_io_info,
      unordered_map<string, Array<Expr> >& _shape,
      unordered_map<string, Type>& _dtype)
  : dev_io_info(_dev_io_info), shape(_shape), dtype(_dtype)  {}

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    string target_name = op->buffer_var.get()->name_hint;
    if (target_name == "test") {
      HCL_DEBUG_LEVEL(2) << "Removed unused var " << target_name;
      return this->Mutate(op->body);
    }
    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::device_scope) {

      // Reconstruct the dev scope function
      if (!op->node.defined()) { 
        Array<Var> undefs = UndefinedVars(op->body, Array<Var>());
        unordered_map<string, IoInfo> dev_io_copy = dev_io_info;

        // Buffers to substitute
        unordered_map<const Variable*, VarExpr> vmap;
        Array<VarExpr>      kernel_def_new_vars;
        Array<Expr>         kernel_stmt_vars;
        vector<Expr>        kernel_stmt_annotate_values;
        Array<Array<Expr> > shapes, attributes; 
        Array<Expr> types;
        Array<FunctionRef> placeholders;

        for (auto& v : undefs) {
          string name = v.get()->name_hint;
          IoInfo io_attr;
          if (!dev_io_copy.count(name)) {
            LOG(INFO) << "Cannot find data placement information "
                << "of tensor " << name << ". Use default setting...";
            io_attr.dev_type      = DeviceType::devFPGA;
            io_attr.storage_type  = StorageType::devDRAM; 
            io_attr.mem_port      = 0;
            io_attr.stream_type   = StreamType::DMA; 
            io_attr.channel_depth = 0;

          } else {
            io_attr = dev_io_copy.at(name);
          }

          CHECK(dtype.count(name) && shape.count(name));
          Type type = dtype[name];
          Array<Expr> arg_shape = shape[name];
          shapes.push_back(arg_shape);
          types.push_back(Type2Expr(type));

          // Prepare function IO attributes
          // Attributes to KernelDef Nodes
          Array<Expr> attr;
          attr.push_back(StringImm::make(name));
          attr.push_back(IntImm::make(Int(32), static_cast<int>(io_attr.storage_type)));
          attr.push_back(IntImm::make(Int(32), io_attr.mem_port));
          attr.push_back(IntImm::make(Int(32), static_cast<int>(io_attr.stream_type)));
          attr.push_back(IntImm::make(Int(32), io_attr.channel_depth));
          attributes.push_back(attr);

          // Create new buffers to replace old buffers
          Var old_var(v.node_);
          VarExpr new_var(name);
          Operation op = PlaceholderOpNode::make(name, arg_shape, type);
          placeholders.push_back(op);
    
          vmap[old_var.get()] = VarExpr(new_var.node_);
          kernel_def_new_vars.push_back(new_var);
          kernel_stmt_vars.push_back(old_var);

          string value = std::to_string(static_cast<int>(io_attr.dev_type)) + ":" +
                std::to_string(static_cast<int>(io_attr.storage_type)) + ":" + 
                std::to_string(io_attr.mem_port) + ":" + 
                std::to_string(static_cast<int>(io_attr.stream_type)) + ":" + 
                std::to_string(io_attr.channel_depth);
          kernel_stmt_annotate_values.push_back(StringImm::make(value));

          dev_io_copy.erase(name);
        }

        // Buffers to be lift atop kernel function call
        unordered_map<string, VarExpr> remove;
        for (auto& kv : dev_io_copy) {
          string name = kv.first;

          CHECK(dtype.count(name) && shape.count(name));
          Type type = dtype[name];
          Array<Expr> arg_shape = shape[name];
          shapes.push_back(arg_shape);
          types.push_back(Type2Expr(type));

          // Prepare function IO attributes
          // attributes entry : Array<name, mem, port, stream, depth, direction>
          Array<Expr> attr;
          auto io_attr = kv.second;
          attr.push_back(StringImm::make(name));
          attr.push_back(IntImm::make(Int(32), static_cast<int>(io_attr.storage_type)));
          attr.push_back(IntImm::make(Int(32), io_attr.mem_port));
          attr.push_back(IntImm::make(Int(32), static_cast<int>(io_attr.stream_type)));
          attr.push_back(IntImm::make(Int(32), io_attr.channel_depth));
          attributes.push_back(attr);

          VarExpr new_var(name);
          remove[name] = new_var;
          kernel_def_new_vars.push_back(new_var);

          Operation op = PlaceholderOpNode::make(name, arg_shape, type);
          placeholders.push_back(op);
        };
        
        // Replace buffers
        SubstituteBuffers sb(vmap, remove);
        Stmt body = sb.Mutate(op->body);

        // Create KernelDef Stmt based on body
        Stmt kernel = KernelDef::make(kernel_def_new_vars, shapes, types, 
                          placeholders, body, UIntImm::make(UInt(1), 1),
                          UInt(32), "test", attributes); 
        kernel_defs_.push_back(kernel);

        // Buffer lifting and return KernelStmt
        CHECK(dev_io_copy.size() == sb.lifted_buffers.size());
        for (auto& var : sb.lifted_buffers) {
            Expr new_arg(var.node_);
            kernel_stmt_vars.push_back(new_arg);

            CHECK(dev_io_copy.count(var.get()->name_hint));
            auto io_attr = dev_io_copy.at(var.get()->name_hint);
            string value = std::to_string(static_cast<int>(io_attr.dev_type)) + ":" +
                  std::to_string(static_cast<int>(io_attr.storage_type)) + ":" + 
                  std::to_string(io_attr.mem_port) + ":" + std::to_string(static_cast<int>(io_attr.stream_type)) + ":" + 
                  std::to_string(io_attr.channel_depth);
            kernel_stmt_annotate_values.push_back(StringImm::make(value));
        }

        // Prepare the annotate keys and values 
        Array<Expr> keys, values;
        for (size_t k = 0; k < kernel_stmt_vars.size(); k++) {
            keys.push_back(IntImm::make(Int(32), k));
            values.push_back(kernel_stmt_annotate_values[k]);
        }
        Stmt stmt = KernelStmt::make(kernel_stmt_vars, "test", keys, values);

        for (auto& var : sb.lifted_buffers) {
          string name = var.get()->name_hint;
          CHECK(dtype.count(name) && shape.count(name));
          Type type = dtype[name];
          Array<Expr> arg_shape = shape[name];
          stmt = Allocate::make(var, type, arg_shape, make_const(Bool(type.lanes()), true), stmt);
          stmt = AttrStmt::make(var, attr::storage_scope, StringImm::make("global"), stmt);
        }
        return stmt;

      }

    }
    return IRMutator::Mutate_(op, s);
  }

  // Replace device scope with KernelStmt
  // and block it with the newly generated KernelDef
  Stmt SplitScope(Stmt stmt) {
    Stmt s = Mutate(stmt);
    for (auto& k : kernel_defs_) s = Block::make(k, s);
    return RemoveNoOp(s);
  }

  unordered_map<string, IoInfo> dev_io_info;
  unordered_map<string, Array<Expr>> shape;
  unordered_map<string, Type> dtype;
  vector<Stmt> kernel_defs_;
};


// Collect the host-device information 
// 1. The IO information that is used to create the KernelDef function 
//    and the KernelStmt for calling the device function  
// 2. Buffer type and shape information 
// 3. Whether a buffer is passed from top level (storage_scope)
class StreamInfoCollector final : public IRMutator {
  public: 
    StreamInfoCollector(Array<NodeRef>& api_args) {
    for (size_t i = 0; i < api_args.size(); i++) { 
      if (const Variable* v = api_args[i].as<Variable>()) {
        top_arg_names.insert(v->name_hint);

      } else if (auto buf = api_args[i].as<BufferNode>()) {
        CHECK(buf->data.as<Variable>());
        top_arg_names.insert(buf->name); 

        shape_[buf->data.get()->name_hint] = buf->shape;
        dtype_[buf->data.get()->name_hint] = buf->dtype;
      }
    }
  };

  Stmt Mutate_(const Partition* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Partition>();
    auto name = op->buffer_var.get()->name_hint;
    name = name.substr(0, name.find(".partitioned")); 
    if (top_arg_names.find(name) != top_arg_names.end()) {
        top_args_partitions[name].push_back(op);
    }
    return stmt;
  }

  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    auto v = op->buffer_var.get();
    auto name = v->name_hint; 
    // Save shape and dtype information
    shape_[name] = op->extents;
    dtype_[name] = op->type;
    return IRMutator::Mutate_(op, s);
  }

  // Record the IO interface information
  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::io_interface) {
      CHECK(op->value.as<StringImm>());
      string s = op->value.as<StringImm>()->value;

      size_t pos = 0;
      string delimiter = ":";
      string token;
      vector<int> numbers;
      while ((pos = s.find(delimiter)) != string::npos) {
          token = s.substr(0, pos);
          numbers.push_back(std::stoi(token));
          s.erase(0, pos + delimiter.length());
      }

      // Memory type, MemPort, StreamType, ChannelDepth
      numbers.push_back(std::stoi(s));
      CHECK(numbers.size() == 6);

      IoInfo io_info;
      io_info.dev_type      = static_cast<DeviceType>(numbers[0]);
      io_info.storage_type  = static_cast<StorageType>(numbers[1]); 
      io_info.mem_port      = numbers[2];
      io_info.stream_type   = static_cast<StreamType>(numbers[3]); 
      io_info.channel_depth = numbers[4];
      io_info.burst_len     = numbers[5];

      VarExpr var(op->node.node_);
      string name = var.get()->name_hint; 
      dev_io_info[name] = io_info;

      return this->Mutate(op->body);

    // The global channel (e.g. Intel channels)
    } else if (op->attr_key == attr::stream_scope) {
      CHECK(op->value.as<StringImm>());
      string s = op->value.as<StringImm>()->value;

      size_t pos = 0;
      string delimiter = ":";
      string token;
      vector<int> numbers;
      while ((pos = s.find(delimiter)) != string::npos) {
          token = s.substr(0, pos);
          numbers.push_back(std::stoi(token));
          s.erase(0, pos + delimiter.length());
      }

      // Channel index, channel depth  
      numbers.push_back(std::stoi(s));
      CHECK(numbers.size() == 2);
      VarExpr var(op->node.node_);
      string name = var.get()->name_hint; 
      global_channel_trace[name] = numbers;
      return this->Mutate(op->body);

    // The tensor to be streamed (inter-stage)
    // Need to create channels explictly  
    } else if (op->attr_key == attr::stream_attrs) {
      CHECK(op->value.as<StringImm>());
      string s = op->value.as<StringImm>()->value;

      size_t pos = 0;
      string delimiter = ":";
      string token;
      vector<int> numbers;
      while ((pos = s.find(delimiter)) != string::npos) {
          token = s.substr(0, pos);
          numbers.push_back(std::stoi(token));
          s.erase(0, pos + delimiter.length());
      }

      // Channel index, channel depth, is_producer 
      numbers.push_back(std::stoi(s));
      CHECK(numbers.size() == 4);
      VarExpr var(op->node.node_);
      string name = var.get()->name_hint; 

      int  channel_index    = numbers[0];
      int  channel_depth    = numbers[1];
      bool is_producer      = (numbers[2] == 1) ? true : false;
      int  max_consumers    = numbers[3];

      // Information processing 
      // 1. If # stream channel <  # consumers. Then 
      //    we need to allocate channels for each streaming pair
      //    and finally written the value back to original buffer
      // 2. If # stream channel == # consumers. Same
      //    case. we do not need to write data back to original buffers
      StreamInfo info;
      if (inter_stage_channels.count(name)) {
        info = inter_stage_channels.at(name);
      }

      if (is_producer) {
        info.index_array.push_back(channel_index);
        info.depth_array.push_back(channel_depth);
        info.max_consumers = max_consumers;
        channel_index *= -1;
      }

      inter_stage_channels[name] = info;
      return AttrStmt::make(op->node, attr::stream_attrs,
            IntImm::make(Int(32), channel_index), this->Mutate(op->body));
    }

    return IRMutator::Mutate_(op, s);
  }

  // Mark the global scoped buffer in KernelStmt 
  Stmt Mutate_(const KernelStmt* op, const Stmt& s) final {
    int pos = 0;
    for (auto arg : op->args) {
      auto name = arg.as<Variable>()->name_hint;
      if (top_arg_names.find(name) != top_arg_names.end())
        global_buffer_trace[op->name].insert(pos);
      pos += 1;
    }
    return IRMutator::Mutate_(op, s);
  }

  unordered_set<string> top_arg_names;
  unordered_map<string, vector<const Partition*> > top_args_partitions;
  unordered_map<string, Array<Expr> > shape_;
  unordered_map<string, Type> dtype_;
  unordered_map<string, IoInfo> dev_io_info;

  unordered_map<string, unordered_set<int> > global_buffer_trace;
  unordered_map<string, vector<int> > global_channel_trace;
  unordered_map<string, StreamInfo>  inter_stage_channels;
};


class StoreToStreamStmtConverter final : public IRMutator {
  public: 
    StoreToStreamStmtConverter(
        const string& target,
        const ir::StreamType& type,
        const VarExpr& channel_buf,
        const int channel_depth,
        int channel_index,
        const Array<Expr> shape,
        unordered_map<const Variable*, Expr>& range) 
      : target_(target), type_(type), channel_buf_(channel_buf),
        channel_depth_(channel_depth), channel_index_(channel_index), 
        shape_(shape), range_(range) {} 

    Stmt Mutate_(const Store* op, const Stmt& s) {
      Expr index = op->index;
      Expr value = this->Mutate(op->value);
      string target_name = op->buffer_var.get()->name_hint;
      if (target_name == target_) {
        Array<Expr> keys, values;
        // push channel and access information 
        keys.push_back(StringImm::make("index"));
        values.push_back(index);
        keys.push_back(StringImm::make("channel"));
        values.push_back(IntImm::make(Int(32), channel_index_));
        return StreamStmt::make(VarExpr(channel_buf_.node_), 0, value, 0,
                                type_, channel_depth_, keys, values); 
      } else {
        return Store::make(op->buffer_var, value, 
                           index, op->predicate);
      }
    }

  private:
    const string target_;
    const ir::StreamType type_;
    const VarExpr& channel_buf_;
    const int channel_depth_;
    const int channel_index_;
    const Array<Expr> shape_;
    unordered_map<const Variable*, Expr>& range_;
};

class LoadToStreamExprConverter final : public IRMutator {
  public: 
    LoadToStreamExprConverter(
        const string& target,
        const ir::StreamType& type,
        const VarExpr& channel_buf,
        const int channel_depth,
        int channel_index, 
        const Array<Expr> shape,
        unordered_map<const Variable*, Expr>& range) 
      : target_(target), type_(type), channel_buf_(channel_buf),
        channel_depth_(channel_depth), channel_index_(channel_index), 
        shape_(shape), range_(range) {} 

    // record axis to mutate streaming sender 
    Stmt Mutate_(const For* op, const Stmt& s) {
      Stmt stmt = IRMutator::Mutate_(op, s);
      if (found) // in the right track
        loop_vars.push_back(op->loop_var.get());
      return stmt;
    }

    // single load repalcement 
    Expr Mutate_(const Load* op, const Expr& e) {
      Expr index = op->index;
      string target_name = op->buffer_var.get()->name_hint;
      if (target_ == target_name) {
        Array<Expr> keys, values;
        // push channel and access information 
        keys.push_back(StringImm::make("index"));
        values.push_back(std::move(op->index));

        keys.push_back(StringImm::make("channel"));
        values.push_back(IntImm::make(Int(32), channel_index_));
        return StreamExpr::make(op->type, VarExpr(channel_buf_.node_), 0, 0,
                                type_, channel_depth_, keys, values);
      } else {
        return Load::make(op->type, op->buffer_var, 
                          index, op->predicate);
      }
   }
    std::vector<const Variable*> loop_vars;

  private:
    bool found{false};           // found tagret load op 
    const string target_;   // stream variable name 
    const ir::StreamType type_;  // stream types (fifo, channel, pipe)
    const VarExpr& channel_buf_; // streaming channel buffer
    const int channel_depth_;    // stream channel depth (no less than 0)
    const int channel_index_;    // stream channel index (share no more than 2 agents)
    const Array<Expr> shape_;    // shape array of target load op
    unordered_map<const Variable*, Expr>& range_; // range map of IterVar
};

class LoadStoreReplacer : public ir::IRMutator {
 public:
  explicit LoadStoreReplacer(
      const std::unordered_map<string, VarExpr>& vsub, bool replace_store)
      : vsub_(vsub), replace_store_(replace_store) {}

    Expr Mutate_(const Load* op, const Expr& e) {
      Expr index = op->index;
      string target_name = op->buffer_var.get()->name_hint;
      if (vsub_.count(target_name) && !replace_store_) {
        VarExpr new_var(vsub_.at(target_name).node_);
        return Load::make(op->type, new_var, index, op->predicate);
      } else {
        return Load::make(op->type, op->buffer_var, index, op->predicate);
      }
    }

    Stmt Mutate_(const Store* op, const Stmt& s) {
      Expr index = op->index;
      Expr value = this->Mutate(op->value);
      string target_name = op->buffer_var.get()->name_hint;
      if (vsub_.count(target_name) && replace_store_) {
        VarExpr new_var(vsub_.at(target_name).node_);
        return Store::make(new_var, value, index, op->predicate);
      } else {
        return Store::make(op->buffer_var, value, index, op->predicate);
      }
    }

 private:
  const unordered_map<string, VarExpr>& vsub_;
  bool replace_store_{false};
};

Stmt BufferInserter(Stmt stmt, const VarExpr var, Array<Expr> shape, /*target buffer shape*/ 
       Type dtype, bool is_write, unordered_map<string, vector<const Partition*> >& top_args_partitions) {

  string name = var.get()->name_hint;

  std::vector<Expr> indices;
  std::vector<VarExpr> loop_vars;
  for (size_t i = 0; i < shape.size(); i++) {
    VarExpr iter(name + ".burst.r" + std::to_string(i));
    indices.push_back(iter);
    loop_vars.push_back(iter);
  }

  Expr index = FlattenIndices(indices, shape); 
  VarExpr new_var(name + ".on.device");
  // Create the burst write at end of the body
  if (is_write) {   

    Expr load = Load::make(dtype, new_var, index, 
        UIntImm::make(UInt(1), 1));
    Stmt for_stmt = Store::make(VarExpr(var.node_), load, index,
        UIntImm::make(UInt(1), 1));
    
    auto type = ForType::Serial;
    for (size_t j = 0; j < shape.size(); j++) {
      auto iter = loop_vars[j];
      // AXI DMA burst store to global memory  
      for_stmt = For::make(VarExpr(iter.node_), 0, shape[j],
        type, DeviceAPI::None, for_stmt);
    }

    // Replace all the store to the new buffer 
    unordered_map<string, VarExpr> vsub;
    vsub[name] = new_var;
    LoadStoreReplacer lsr(vsub, true);
    stmt = lsr.Mutate(stmt);

    stmt = Block::make(stmt, for_stmt); 
    // Add allocate IR node
    Array<Stmt> attrs;
    if (top_args_partitions.count(name)) {
      for (auto& op : top_args_partitions.at(name)) {
        auto ps = Partition::make(new_var, op->dim, op->factor, op->partition_type);
        attrs.push_back(ps);
      }
      top_args_partitions.erase(name);
    }

    stmt = Allocate::make(new_var, dtype, shape, 
               make_const(Bool(dtype.lanes()), true), stmt, attrs, Expr(), string());
    stmt = AttrStmt::make(new_var, attr::storage_scope, StringImm::make("global"), stmt);

  // Create read burst at begining of the body
  } else {  
    // Load from original buffers
    Expr load = Load::make(dtype, VarExpr(var.node_), index, 
        UIntImm::make(UInt(1), 1));
    Stmt for_stmt = Store::make(new_var, load, index,
        UIntImm::make(UInt(1), 1));

    auto type = ForType::Serial;
    for (size_t j = 0; j < shape.size(); j++) {
      auto iter = loop_vars[j];
      // AXI DMA burst load from global memory  
      for_stmt = For::make(VarExpr(iter.node_), 0, shape[j],
        type, DeviceAPI::None, for_stmt);
    }

    // Replace all the store to the new buffer 
    unordered_map<string, VarExpr> vsub;
    vsub[name] = new_var;
    LoadStoreReplacer lsr(vsub, false);
    stmt = lsr.Mutate(stmt);

    stmt = Block::make(for_stmt, stmt); 
    // Add allocate IR node
    Array<Stmt> attrs;
    if (top_args_partitions.count(name)) {
      for (auto& op : top_args_partitions.at(name)) {
        auto ps = Partition::make(new_var, op->dim, op->factor, op->partition_type);
        attrs.push_back(ps);
      }
      top_args_partitions.erase(name);
    }
    stmt = Allocate::make(new_var, dtype, shape, 
               make_const(Bool(dtype.lanes()), true), stmt, attrs, Expr(), string());
    stmt = AttrStmt::make(new_var, attr::storage_scope, StringImm::make("global"), stmt);

  }

  return stmt;
};

// create streaming channels across loop iterations
class LoopbackMutator : public ir::IRMutator {
 public:
  explicit LoopbackMutator(
    const VarExpr& target_buf, const Array<Expr>& shape,
    const unordered_map<const Variable*, Expr>& range, 
    Type type)
  : target_buf_(target_buf), shape_(shape), 
    range_(range), type_(type) {} 

  // FIXME: buffer mismatch 
  Stmt Mutate_(const Store* op, const Stmt& s) {
    if (op->buffer_var->name_hint == target_buf_->name_hint) {
      if (store_count == 0) { 
        store_count += 1;
        CHECK(!temp_.defined());
        temp_ = VarExpr("temp_" + target_buf_->name_hint); 
        auto index = IntImm::make(Int(32), 0);
        Expr load_expr = Load::make(type_, 
                             temp_, index, op->predicate);
        save_stmt = Store::make(op->buffer_var, 
                        load_expr, op->index, op->predicate);

        Stmt stmt = Store::make(temp_, op->value, index, op->predicate);
        stmt = Allocate::make(temp_, type_, Array<Expr>(),
            make_const(Bool(type_.lanes()), true), stmt);
        stmt = AttrStmt::make(temp_, attr::storage_scope,
            StringImm::make("local"), stmt);
        return stmt;

      } else {
        store_count += 1;
        auto index = IntImm::make(Int(32), 0);
        return Store::make(temp_, op->value, index, op->predicate);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    if (op->buffer_var->name_hint == target_buf_->name_hint) {
      if (store_count > 0) { 
        auto index = IntImm::make(Int(32), 0);
        return Load::make(op->type, temp_, index, op->predicate);
      }
    }
    return e;
  }

  // create stream array
  Stmt Mutate_(const For* op, const Stmt& s) {

    if (op->body.as<For>() == nullptr) {
      Stmt stmt = this->Mutate(op->body);
      stmt = Block::make(stmt, save_stmt);
      return For::make(
          op->loop_var, op->min, op->extent, op->for_type,
          op->device_api, stmt, op->annotate_keys,
          op->annotate_values);

    } else {
      Stmt stmt = this->Mutate(op->body);
      return For::make(
          op->loop_var, op->min, op->extent, op->for_type,
          op->device_api, stmt, op->annotate_keys,
          op->annotate_values);
    }
  }

  private:
   const VarExpr& target_buf_;
   const Array<Expr>& shape_;
   const unordered_map<const Variable*, Expr>& range_;
   Type type_; 
   VarExpr temp_;
   int store_count{0};
   Stmt save_stmt;
};


// create local copy and sync with data copy 
class MultiLoadMutator : public IRMutator {
 public:
  explicit MultiLoadMutator(
    string& target,
    std::vector<VarExpr>& channels, Type type)
    : target_(target), channels_(channels), type_(type) {}

  Stmt Mutate(Stmt stmt) final {
    Stmt ret = IRMutator::Mutate(stmt);
    if (found && !alloc) { 
      for (auto& channel : channels_) {
        auto stream_expr = StreamExpr::make(type_, 
            VarExpr(channel.node_), 0, 0, StreamType::FIFO, 
            1, Array<Expr>(), Array<Expr>()); 

        auto store = Store::make(temp_, 
                stream_expr, Expr(0), const_true());
        ret = Block::make(store, ret);
      }
      ret = Allocate::make(temp_, type_, Array<Expr>(),
          make_const(Bool(type_.lanes()), true), ret);
      ret = AttrStmt::make(temp_, attr::storage_scope,
          StringImm::make("local"), ret);
      alloc = true;
    }
    return ret;
  }

  Expr Mutate_(const Load *op, const Expr& e) final {
    Expr index = op->index;
    string target_name = op->buffer_var.get()->name_hint;

    Stmt stmt;
    if (target_name == target_) {
      found = true;
      temp_ = VarExpr("temp_" + target_);
      return Load::make(op->type, temp_, index, op->predicate);
    } else {
      return Load::make(op->type, op->buffer_var, index, op->predicate);
    }
  }

 private:
  string& target_;
  std::vector<VarExpr>& channels_;
  Type type_;
  VarExpr temp_;
  bool found{false};
  bool alloc{false};
};

// Collect access pattern for kernel function args
class AccessCollector : public ir::IRMutator {
 public:
  explicit AccessCollector(
      const VarExpr& target_buf, const Array<Expr>& shape,
      const unordered_map<const Variable*, Expr>& range,
      const string channel_name)
      : target_buf_(target_buf), shape_(shape), range_(range), 
        channel_name_(channel_name) {}

  // trace buffer allocation 
  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    string target_name = op->buffer_var.get()->name_hint;
    // whether the target buffer has been allocated 
    if (target_name == channel_name_) buf_alloc = true;
    return s;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    Expr value = this->Mutate(op->value);
    string target_name = op->buffer_var.get()->name_hint;
    // check if target buffer matches
    if (op->buffer_var.get() == target_buf_.get()) {
      store_num += 1; 
      store_var = VarExpr(op->buffer_var.node_);
      // check index access regularity 
      auto max_bound = Substitute(op->index, range_); 
      reg_store = is_zero(Simplify(max_bound - get_max(shape_)));
    }
    return s;
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    string target_name = op->buffer_var.get()->name_hint;
    // check if buffer matches
    if (op->buffer_var.get() == target_buf_.get()) {
      load_num += 1; 
      load_var = VarExpr(op->buffer_var.node_);
      // check index access regularity 
      auto max_bound = Substitute(op->index, range_); 
      reg_load = is_zero(Simplify(max_bound - get_max(shape_)));
    }
    return e;
  }

  int load_num{0};
  int store_num{0};
  VarExpr load_var;
  VarExpr store_var;
  bool reg_store{true};
  bool reg_load{true};
  bool buf_alloc{false};

 private:
  const VarExpr& target_buf_; /*stream variable buffer*/ 
  const Array<Expr>& shape_;  /*stream variable shape*/ 
  const unordered_map<const Variable*, Expr>& range_;
  const string channel_name_;

  Expr get_max(Array<Expr> shape) {
    Expr ret(shape[0]); 
    for (size_t i = 1; i < shape.size(); i++) ret *= shape[i]; 
    return Simplify(ret - 1); 
  }
};

// Detect the direction of input args 
class InputDirectionCollector : public ir::IRMutator {
 public:
  explicit InputDirectionCollector(
    const Array<VarExpr>& _input_vars) : input_vars(_input_vars) { }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    if (checkBuffer(op->buffer_var)) {
        is_arg_written[op->buffer_var.get()->name_hint] = true;
    }
    return stmt;
  }

  bool checkBuffer(VarExpr var) {
    for (auto& v : input_vars) {
      if (v.get() == var.get()) {
        HCL_DEBUG_LEVEL(2) << "[ INFO ] Buffer arg "
            << v.get()->name_hint << " is written...";
        return true;
      }
    }
    return false;
  }

  unordered_map<string, bool>& Analyze(Stmt stmt) {
    for (auto& v : input_vars) {
      is_arg_written[v.get()->name_hint] = false;
    }
    Stmt s = Mutate(stmt);
    return is_arg_written; 
  }

  const Array<VarExpr>& input_vars;
  unordered_map<string, bool> is_arg_written;
};

// Attribute non-kernel definitions
class KernelDefDecorator final : public IRMutator {
 public: 
  Stmt Mutate_(const KernelDef* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<KernelDef>();
    Array<VarExpr> input_args;
    for (auto& v : op->args) {
        input_args.push_back(v);
    }
    InputDirectionCollector idc(input_args);
    auto is_arg_written = idc.Analyze(op->body);

    Array<Array<Expr> > new_attrs;
    // Top-level kernel function
    if (op->attributes.size() == op->args.size()) {
        for (auto& attr : op->attributes) {
            CHECK(attr.size() > 0);
            auto name = attr[0].as<StringImm>();
            CHECK(name);
            string arg_name = name->value;

            CHECK(is_arg_written.count(arg_name)) << op->attributes;
            Array<Expr> new_attr = attr;
            if (is_arg_written.at(arg_name)) {
                Expr direction = IntImm::make(Int(32), 1);
                new_attr.push_back(direction);
            } else {
                Expr direction = IntImm::make(Int(32), 0);
                new_attr.push_back(direction);
            }
            new_attrs.push_back(new_attr);
        }

    } else {

        for (auto& v : op->args) {
            auto arg_name = v.get()->name_hint;
            CHECK(is_arg_written.count(arg_name)) << arg_name;
            Array<Expr> new_attr = { StringImm::make(arg_name) };
            if (is_arg_written.at(arg_name)) {
                Expr direction = IntImm::make(Int(32), 1);
                new_attr.push_back(direction);
            } else {
                Expr direction = IntImm::make(Int(32), 0);
                new_attr.push_back(direction);
            }
            new_attrs.push_back(new_attr);
        }

        // Process the streaming informatin
        for (auto& attr : op->attributes) {
            CHECK(attr.size() > 0);
            HCL_DEBUG_LEVEL(2) << " -- Processing kernel streaming arg " << attr[0] << "...";
        }
    }

    HCL_DEBUG_LEVEL(2) << " -- New attrs for kernel " << op->name << "\n" << new_attrs;
    return KernelDef::make(op->args, op->arg_shapes, op->arg_types, op->arg_tensors,
                           op->body, op->ret_void, op->ret_type, op->name, new_attrs);
  }
};

// Create write or read loops in the top kernel def
class CreateBurstLoops final : public IRMutator {
 public: 
  CreateBurstLoops(unordered_map<string, IoInfo>& _dev_io_info,
      unordered_map<string, Array<Expr> >& _shape,
      unordered_map<string, Type>& _dtype,
      unordered_map<string, vector<const Partition*> > _top_args_partitions)
  : dev_io_info(_dev_io_info), shape(_shape), dtype(_dtype),
    top_args_partitions(_top_args_partitions)  {}

  Stmt Mutate_(const KernelDef* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<KernelDef>();
    Stmt body = op->body;

    // Top-level kernel function
    if (op->attributes.size() == op->args.size()) {
        size_t index = 0;
        for (auto& arg : op->args) {
            auto name = arg.as<Variable>()->name_hint;
            auto attrs = op->attributes[index];
            auto size = attrs.size();
            CHECK(attrs[size-1].as<IntImm>());
            auto direction = attrs[size-1].as<IntImm>()->value;
            bool write = (direction == 1) ? true : false;

            if (burst_arg_list.count(name)) {
                string write_ = (write) ? "write" : "read";
                HCL_DEBUG_LEVEL(2) << "[ debug ] Insert " << write_ 
                    << " burst for " << name;
                auto shape = op->arg_shapes[index];
                CHECK(dtype.count(name));
                auto type  = dtype[name];
                body = BufferInserter(body, arg, shape, type, write, top_args_partitions);
            }
            index ++;
        }
    }

    if (top_args_partitions.size() > 0) {
        for (auto& kv : top_args_partitions) {
            for (auto& op : kv.second) {
                HCL_DEBUG_LEVEL(2) << "[ debug ] insert placeholder partition stmts " << op->buffer_var;
                auto ps = Partition::make(op->buffer_var, op->dim, op->factor, op->partition_type);
                body = Block::make(ps, body);
            }
        }
    }
    return KernelDef::make(op->args, op->arg_shapes, op->arg_types, op->arg_tensors,
                           body, op->ret_void, op->ret_type, op->name, op->attributes);
  }

  Stmt Insert(Stmt stmt) {
    for (auto& info : dev_io_info) {
        if (info.second.burst_len >=0) {
            burst_arg_list[info.first] = info.second.burst_len;
        }
    }
    if (burst_arg_list.size() == 0) {
        return stmt;
    }
    Stmt s = Mutate(stmt);
    return RemoveNoOp(s);
  }

  unordered_map<string, IoInfo> dev_io_info;
  unordered_map<string, Array<Expr>> shape;
  unordered_map<string, Type> dtype;
  unordered_map<string, int> burst_arg_list;
  unordered_map<string, vector<const Partition*> > top_args_partitions;
};


class CreateSelfLoopBackChs final : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    // Collect information for self-streaming channels
    if (op->attr_key == attr::device_scope) {
        VarExpr var(op->node.node_);
        auto name = var.get()->name_hint;
        CHECK(op->value.as<IntImm>()) << op->value;
        auto depth = op->value.as<IntImm>()->value;
        stream_ch_maps[name] = depth;     
        HCL_DEBUG_LEVEL(2) << "[debug] Creating FIFO attr for tensor " << name << "...";
        return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();

    string name = op->buffer_var.get()->name_hint;
    if (stream_ch_maps.count(name)) {

      HCL_DEBUG_LEVEL(2) << "[debug] Found Streaming Channel " << name;
      int channel_depth = stream_ch_maps.at(name);
      Stmt attr = StreamStmt::make(
            op->buffer_var, 0, IntImm::make(Int(32), 0), 0,
            StreamType::ATTR, channel_depth, Array<Expr>(), Array<Expr>()); 
      Array<Stmt> attrs = op->attrs;
      attrs.push_back(attr);
      return Allocate::make(op->buffer_var, op->type, op->extents,
                            op->condition, op->body, attrs,
                            op->new_expr, op->free_function);
    }
    return stmt;
  }

  unordered_map<string, int> stream_ch_maps;
};


class FifoAccessChecker final : public IRMutator {

 private:
  struct FifoInfo {
    int   depth{0};
    int   consumers{0};
    int   producers{0};
    int   read_bound{0};
    int   write_bound{0};
  };
 public:

  Stmt Mutate_(const Allocate* op, const Stmt& s) {

    for (auto attr : op->attrs) {
      if (auto ss = attr.as<StreamStmt>()) {
        HCL_DEBUG_LEVEL(2) << "[ INFO ] Created FIFO " << op->buffer_var << " (depth="
          << ss->depth << ")";
        string name = op->buffer_var.get()->name_hint;
        FifoInfo fifo_info;
        fifo_info.depth = ss->depth;
        fifo_info_map[name] = fifo_info;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  // Ignore the nest loops after allocate node
  Stmt Mutate_(const For* op, const Stmt& s) {
    min_map_[op->loop_var.get()] = op->min;
    range_[op->loop_var.get()] = op->extent - 1;
    Stmt stmt = op->body;
    while (const For* for_op = stmt.as<For>()) {
      stmt = for_op->body;
    }
    if (auto st = stmt.as<Store>()) {
      auto value = st->value;
      if (auto c = value.as<Cast>()) value = c->value;
      if (auto v = value.as<IntImm>()) {
        if (v->value == 0) return s;
      } else if (auto v = value.as<FloatImm>()) {
        if (v->value == 0) return s;
      } else if (auto v = value.as<UIntImm>()) {
        if (v->value == 0) return s;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    string name = op->buffer_var.get()->name_hint;

    if (fifo_info_map.count(name)) {
      auto& info = fifo_info_map.at(name);
      info.producers += 1;
      CHECK(info.producers == 1) << "FIFO " << name << " produced multiple times...";
      CHECK(info.consumers <= 1) << "FIFO " << name << " consumed multiple times...";
      HCL_DEBUG_LEVEL(2) << "[ INFO ] FIFO write " << name << " found. Convert to StreamStmt..."; 

      // Check the access bound
      Expr max = Simplify(substitute(range_, op->index));
      Expr min = Simplify(substitute(min_map_, op->index));
      Expr num = Simplify(max - min);
      CHECK(num.as<IntImm>()) << "FIFO max write times " << num << " not a int value";
      info.write_bound = num.as<IntImm>()->value;

      // Connvert to StreamStmt
      return StreamStmt::make(op->buffer_var, op->index, op->value, 0,
              StreamType::FIFO, info.depth, Array<Expr>(), Array<Expr>()); 
    }
    auto value = this->Mutate(op->value);
    return Store::make(op->buffer_var, value, op->index, op->predicate);
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    string name = op->buffer_var.get()->name_hint;
    if (fifo_info_map.count(name)) {
      auto& info = fifo_info_map.at(name);
      info.consumers += 1;
      CHECK(info.producers == 1) << "FIFO " << name << " produced multiple times...";
      CHECK(info.consumers == 1) << "FIFO " << name << " consumed multiple times...";
      HCL_DEBUG_LEVEL(2) << "[ INFO ] FIFO read " << name << " found. Convert to StreamExpr..."; 

      // Check the access bound
      Expr max = Simplify(substitute(range_, op->index));
      Expr min = Simplify(substitute(min_map_, op->index));
      Expr num = Simplify(max - min);
      CHECK(num.as<IntImm>()) << "FIFO max read times " << num << " not a int value";
      info.read_bound = num.as<IntImm>()->value;
      if (info.read_bound != info.write_bound) {
        HCL_DEBUG_LEVEL(2) << "[WARNING] FIFO " << name << " read " << info.read_bound
          << " while write " << info.write_bound << "times...";
      }
      HCL_DEBUG_LEVEL(2) << "[ INFO ] Checking passed: FIFO " << name << " access time " << info.read_bound;
      // Connvert to StreamExpr
      return StreamExpr::make(op->type, op->buffer_var, op->index, 0,
              StreamType::FIFO, info.depth, Array<Expr>(), Array<Expr>()); 
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Convert(Stmt s) {
    return this->Mutate(s);
  }

  unordered_map<string, FifoInfo> fifo_info_map;
  std::map<const Variable*, Expr> range_;
  std::map<const Variable*, Expr> min_map_;
};


Stmt InferStream(Stmt stmt, Array<NodeRef> api_args) {

  // Parse the IO interface information
  StreamInfoCollector sic(api_args);
  stmt = sic.Mutate(stmt);

  // If any inter-stage or inter-module varibles, 
  // 1. insert StreamStmt into its attr scope
  // 2. Create streaming channels (explicitly) for inter-stage 
  AllocateAttrDecorator aad(sic.global_channel_trace,
      sic.inter_stage_channels, sic.dtype_, sic.shape_);
  stmt = aad.Mutate(stmt);

  // Mutate the device_scope AttrStmt 
  KernelDefCreator kdc(sic.dev_io_info, sic.shape_, sic.dtype_);
  stmt = kdc.SplitScope(stmt);

  // Remove the unused buffers
  UnusedBufferRemover ubr(aad.unused_write_buffers);
  stmt = ubr.Mutate(stmt);

  // Add attributes for non-kernel functions definition
  stmt = KernelDefDecorator().Mutate(stmt);

  // Create busrt read or write loop
  CreateBurstLoops cb(sic.dev_io_info, sic.shape_, sic.dtype_, sic.top_args_partitions);
  stmt = cb.Insert(stmt);

  // Handle self loopback streaming channels
  CreateSelfLoopBackChs csfb;
  stmt = csfb.Mutate(stmt);

  // Perform FIFO access order checking 
  // Convert read and write ops into StreamStmt and StramExpr
  HCL_DEBUG_LEVEL(2) << stmt;
  FifoAccessChecker fac;
  stmt = fac.Convert(stmt);

  return stmt;
}

}  // namespace ir
}  // namespace TVM
