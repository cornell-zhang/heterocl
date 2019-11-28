/*!
 *  Copyright (c) 2019 by Contributors
 * \file build_common.cc
 * \brief Build unified simulation module
 */
#include <tvm/base.h>
#include <tvm/ir_visitor.h>
#include <tvm/runtime/config.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/build_module.h>
#include "./build_common.h"
#include "./build_helper.h"

#include <fstream>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>

#include "merlinc/codeanalys_merlinc.h"
#include "hlsc/codegen_vhls.h"
#include "opencl/codegen_aocl.h"
#include "ppac/codegen_rv64_ppac.h"

namespace TVM {
namespace runtime {

class SimModuleNode final : public ModuleNode {
 public:
  SimModuleNode(LoweredFunc func, 
                std::string host_code,
                std::vector<std::tuple<bool, Type, std::vector<int>>> arg_stream_types,
                std::string dev_code, std::string platform, std::unordered_map<std::string, std::string> options)
    : func_(func), 
      host_(host_code), 
      arg_stream_types_(arg_stream_types),
      dev_(dev_code), platform_(platform), options_(options) { 
  }

  const char* type_key() const {
    return "unified_sim";
  }

  // unified simulation function
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv){
        
        if (args.size() != (int)func_->args.size())
          LOG(FATAL) << "The function should take in " << func_->args.size() 
                     << " inputs but get " << args.size();
        std::vector<int> shmids;
        std::vector<size_t> arg_sizes;
        std::vector<TVMType> arg_types;

        CollectArgInfo(args, func_, arg_sizes, arg_types);
        GenSharedMem(args, shmids, arg_sizes);

        LOG(CLEAN) << "Generating harness files ...";
        system("rm -rf __tmp__; mkdir __tmp__");
        std::string path; 
        if (const auto* f = Registry::Get("get_util_path")) 
          path = (*f)(platform_).operator std::string();
        system(("cp -r " + path + "/* __tmp__/").c_str());

        if (platform_ == "sdaccel") {
          GenWrapperCode(args, shmids, arg_types, arg_stream_types_, func_);
          GenHostCode(args, shmids, arg_types, func_, 
                      host_, arg_stream_types_);
          GenKernelCode(dev_);

          LOG(CLEAN) << "Running SW simulation ...";
          system("cd __tmp__; source ./run_sw.sh");

        } else if (platform_ == "rocket") {
          // generate host and run proxy kernel test 
          GenHostCode(args, shmids, arg_types, func_, 
                      host_, arg_stream_types_);
          std::string compile = "cd __tmp__;";
          compile += std::string("autoconf; mkdir build; cd build;") +
                     std::string("../configure --with-riscvtools=") + 
                     options_["RISCV"] + std::string(";make -j8");
          system(compile.c_str());

        } else if (platform_ == "vivado_hls") {
          GenHostCode(args, shmids, arg_types, func_, 
                      host_, arg_stream_types_);
          GenKernelCode(dev_);
        } else {
          LOG(FATAL) << "unrecognized platform " << platform_;  
        } 

        // clean & extract resource information
        FreeSharedMem(args, shmids, arg_sizes);
        if (const auto* f = Registry::Get("tvm_callback_syn_postproc")) {
          std::string code;
          code = (*f)("test").operator std::string();
          LOG(CLEAN) << "extract res info";
        }
      });
  }

 private:
  LoweredFunc func_;
  std::string host_;
  std::vector<std::tuple<bool, Type, std::vector<int>>> arg_stream_types_;
  std::string dev_;
  std::string platform_;
  std::unordered_map<std::string, std::string> options_;
};

using var2nameType = std::unordered_map<const Variable*, 
    std::tuple<std::string, Type, std::vector<int>>>; 

Module CreateSimModule(
    LoweredFunc func,
    std::string host_code,
    std::string dev_code,
    std::vector<std::tuple<bool, Type, std::vector<int>>> arg_type,
    std::string platform, std::unordered_map<std::string, std::string> options) {
  std::shared_ptr<SimModuleNode> n =
    std::make_shared<SimModuleNode>(func, host_code, 
                                    arg_type, dev_code,
                                    platform, options);
  return Module(n);
}
} // namespace runtime

namespace codegen {
using var2nameType = std::unordered_map<const Variable*, 
    std::tuple<std::string, Type, std::vector<int>>>; 

// collect type info for vars
class TypeCollector final : public IRVisitor {
  public:
    var2nameType& top_args_;
    TypeCollector(var2nameType& top_args)
      : top_args_(top_args) {}
    void Visit_(const Allocate *op) {
      auto v = op->buffer_var.get();
      
      // record type and shape
      if (top_args_.count(v)) {
        std::vector<int> shape;
        for (size_t i = 0; i < op->extents.size(); i++) 
          shape.push_back(op->extents[i].as<IntImm>()->value);
        top_args_[v] = std::make_tuple(
                           std::get<0>(top_args_[v]),
                           op->type, shape);
      }
      IRVisitor::Visit_(op);
    }
};

// record <name, type> of vars for top func signature
// vars include passed-in and not registered vars on host
class StreamCollector final : public IRVisitor {
  public:
    StreamCollector(std::vector<const Variable*>& arg_vars,
                    std::unordered_map<const Variable*, bool>& stream_table,
                    std::string initial_scope)
      : arg_vars_(arg_vars),
        stream_table_(stream_table),
        scope_(initial_scope) {}

    // record alloc on host 
    void Visit_(const Allocate *op) {
      if (!switch_on) 
        this->HandleDef(op->buffer_var.get());
      IRVisitor::Visit_(op);
    }
    
    void Visit_(const Load *op) {
      if (!switch_on) {
        this->HandleUse(op->buffer_var);
      }
      IRVisitor::Visit_(op);
    }

    // update placeholder status
    void Visit_(const Store* op) {
      if (switch_on) {
        if (auto val = op->value.as<StreamExpr>()) {
          const Variable* v = val->buffer_var.get();
          for (size_t i = 0; i < arg_vars_.size(); i++) {
            std::string name = arg_vars_[i]->name_hint;
            if (v->name_hint.find(name) != std::string::npos) {
              // record in VisitStmt StreamStmt
              // LOG(WARNING) << op->buffer_var << ":" << v->name_hint;
            }
          }
        }
      } else { // count use on host
        this->HandleUse(op->buffer_var);
      }
      IRVisitor::Visit_(op);
    }

    void Visit_(const StreamStmt* op) {
      if (switch_on) { // in xcel scope
        const Variable* v = op->buffer_var.get();
        // LOG(WARNING) << v->name_hint;  
      }
      IRVisitor::Visit_(op);
    }

    void Visit_(const AttrStmt* op) {
      if (op->attr_key == attr::device_scope) { 
        if (op->value.as<StringImm>()->value != scope_)
          switch_on = true;
        else switch_on = false;
      }
      IRVisitor::Visit_(op);
    }

    // additional data saved into stream table (for streamed 
    // data we keep the new id for arg_stream in var_idmap, 
    // and non-streamed using the repalced arg_top_k name)
    void HandleDef(const Variable* v) {
      CHECK(!host_def_count_.count(v))
          << "variable " << v->name_hint
          << " has already been defined, the Stmt is not SSA";
      CHECK(!host_use_count_.count(v))
          << "variable " << v->name_hint
          << " has been used before definition!";
      host_use_count_[v] = 0;
      host_def_count_[v] = 1;
    }

    void HandleUse(const Expr& v) {
      CHECK(v.as<Variable>());
      Var var(v.node_);
      auto it = host_use_count_.find(var.get());
      if (it != host_use_count_.end()) {
        if (it->second >= 0) {
          ++it->second;
        }
      } else {
        if (!stream_table_.count(var.get())) {
          host_undefined_.push_back(var);
          host_use_count_[var.get()] = -1;
        }
      }
    }

    bool host_scope_{false};
    Array<Var> host_undefined_;
    std::unordered_map<const Variable*, int> host_use_count_;
    std::unordered_map<const Variable*, int> host_def_count_;

  private:
    std::vector<const Variable*>& arg_vars_;
    std::unordered_map<const Variable*, bool>& stream_table_;
    std::string scope_;
    bool switch_on{true};
};

// codegen for accelerators 
class CodeGenXcel : public CodeGenVivadoHLS {
  public:
    int arg_top_count{0};
    str2tupleMap<std::string, Type> map_arg_type_;
    LoweredFunc f_;

  void AddFunction(LoweredFunc f,
           str2tupleMap<std::string, Type> map_arg_type) {
    map_arg_type_ = map_arg_type; f_ = f;
    CodeGenVivadoHLS::AddFunction(f, map_arg_type);
  };

  void VisitStmt_(const AttrStmt* op) {
     if (op->attr_key == ir::attr::device_scope) {
      // print top( ... in host and enter fpga scope 
      if (op->value.as<StringImm>()->value == "fpga" && !fpga_scope_) {
        fpga_scope_ = true;
        PrintIndent();
         
        // track the stream usage
        StreamCollector collector(arg_vars, stream_table, "cpu");
        collector.Visit(op->body);

        // update data type and name 
        for (auto k : collector.host_undefined_) {
          auto v = k.get();
          arg_vars.push_back(v);
          stream_table[v] = true;
          auto tuple = arg_top_vars[v];
          arg_top_vars[v] = std::make_tuple(v->name_hint,
                                            std::get<1>(tuple),
                                            std::get<2>(tuple)); 
        }
        TypeCollector visitor(arg_top_vars);
        visitor.Visit(op->body);
  
        // generte function calls 
        stream << "top(";
        int index = 0;
        for (size_t i = 0; i < arg_vars.size(); i++) {
          auto v = arg_vars[i];
          std::string arg_name;
          if (stream_table[v]) 
            arg_name = std::get<0>(arg_top_vars[v]);
          else arg_name = GetVarID(v); 
          if (index !=0) stream << ", ";
          stream << arg_name;
          // print kernel func signature
          if (index !=0) arg_stream << ", ";
          PrintType(std::get<1>(arg_top_vars[v]), arg_stream);
          auto shape = std::get<2>(arg_top_vars[v]);
          arg_stream << " " << arg_name;
          for (size_t k = 0; k < shape.size(); k++)
            arg_stream << "[" << shape[k] << "]";
          index++;
        }
        stream << ");\n";
  
        // switch context to device scope
        host_stream << this->stream.str();
        this->stream.str("");
        this->stream.clear();
  
      // swtich from device to host
      } else if (op->value.as<StringImm>()->value == "cpu" && 
                 fpga_scope_) {
        fpga_scope_ = false;
        device_stream << this->stream.str();
        this->stream.str("");
        this->stream.clear();
      }
    }
    CodeGenC::VisitStmt_(op);
  }
    void VisitStmt_(const Store* op) {
      std::string vid = GetVarID(op->buffer_var.get());
      if (vid.find("stream_") == std::string::npos)
        CodeGenVivadoHLS::VisitStmt_(op);
    };

    void VisitStmt_(const LetStmt* op) {
      std::string value = PrintExpr(op->value);
      // Skip the argument retrieving assign statement
      std::string vid = AllocVarID(op->var.get());
      if (op->var.type() != Handle() &&
          value.find("TVMArray") == std::string::npos &&
          value.find("arg") != 0) {
        PrintIndent();
        PrintType(op->var.type(), this->stream);
        this->stream << ' '
                     << vid
                     << " = " << value << ";\n";
      // modify var idmap for passed in args
      } else if (value.find("data") != std::string::npos ||
                 value.substr(0, 3) == "arg") {
        auto v = op->var.get();
        auto tuple = arg_top_vars[v]; 
        arg_vars.push_back(v);
        stream_table[v] = false;
        var_idmap_[v] = "arg_top_" + std::to_string(arg_top_count);
        std::string api_name = "arg" + std::to_string(arg_top_count);
        auto arg = map_arg_type_[api_name];
        // PrintType(std::get<1>(arg), arg_stream);
        std::vector<int> shape;
        if (auto buf = f_->api_args[arg_top_count].as<BufferNode>())
          for (size_t i = 0; i < buf->shape.size(); i++) 
            shape.push_back(buf->shape[i].as<IntImm>()->value);
        arg_top_vars[v] = std::make_tuple(vid, std::get<1>(arg), shape);
        arg_top_count += 1;
      }
      PrintStmt(op->body);
    };

    void VisitStmt_(const StreamStmt* op) {
      //TODO: fix this
      // std::string vid = GetVarID(op->buffer_var.get());
      std::string vid;
      if (!var_idmap_.count(op->buffer_var.get())) 
        vid = AllocVarID(op->buffer_var.get());
      else vid = GetVarID(op->buffer_var.get());
      PrintIndent();
      auto load_op = op->value.as<Load>(); 
      auto v = load_op->buffer_var.as<Variable>();
      // placeholder args using recv name 
      if (stream_table.count(v)) {
        auto tuple = arg_top_vars[v];
        vid.replace(vid.find("stream_send"), 12, "stream_recv");
        arg_top_vars[v] = std::make_tuple(vid, std::get<1>(tuple),
                                          std::get<2>(tuple));
        stream_table[v] = true;
      } // else: streamed externop defined in analysis
      // PrintExpr(op->value, stream);
      // stream << vid << ".write()\n";
    };

    void VisitStmt_(const Allocate* op) {
      std::string vid = AllocVarID(op->buffer_var.get());
      CHECK(!is_zero(op->condition));
      int32_t constant_size = op->constant_allocation_size();
      CHECK_GT(constant_size, 0)
          << "Can only handle constant size stack allocation for now";
      const Variable* buffer = op->buffer_var.as<Variable>();
      var_shape_map_[buffer] = op->extents;
      std::string scope = alloc_storage_scope_.at(buffer);
      PrintStorageScope(scope, stream);

      // initlize hls stream channel
      if (arg_top_vars.count(buffer) ||
          vid.find("stream_") != std::string::npos) { 
      } else {
        this->PrintIndent();
        PrintType(op->type, stream);
        stream << ' '<< vid;
        if (constant_size > 1) {// Transfer length one array to scalar
          for (size_t i = 0; i < op->extents.size(); i++) {
            stream << '[';
            PrintExpr(op->extents[i], stream);
            stream << "]";
          }
        }
        stream << ";\n";
      }
      buf_length_map_[buffer] = constant_size;
      RegisterHandleType(op->buffer_var.get(), op->type);
      for (size_t i = 0; i < op->attrs.size(); i++) {
        this->PrintStmt(op->attrs[i]);
      }
      this->PrintStmt(op->body);
    };
};

// replace host-device interface args with pragma 
class CodeGenHost : public CodeGenAOCL {
  public:
    int arg_top_count{0};

  void PrintType(Type t, std::ostream &os) {
    int lanes = t.lanes();
    
    if(t.is_handle())
    {
      os << "void*";return;
    }
    if(t==Bool())
    {
      os <<"bool"; return;
    }
    CHECK_EQ(lanes,1)
        << "do not yet support vector types";
    
    bool fail = false;
    if(t.is_float())
    {
      switch(t.bits())
      {
        case 16:
          os<<"half";
          // enable_fp16_ = true;
          break;
        case 32:
          os<<"float";
          break;
        case 64:
          os<< "double";
          // enable_fp64_ = true;
          break;
        default:
          fail = true;
          break;
      }
      if(!fail && lanes ==1)return;
      if(!fail&&(lanes >= 2 && lanes <=16))
      {
        os<<lanes; return;
      }
    }
    else if(t.is_uint()||t.is_int())
    {
      switch(t.bits())
      {
        case 8: os<< "char"; break;
        case 16: os<<"short"; break;
        case 32: 
          if(t.is_uint())
            os<<"u";
          os<<"int";
          break;
        case 64: os<<"long";break;
        default : fail = true;break;
      }
      if(!fail && lanes == 1)return;
      if(!fail && (lanes >=2 && lanes <= 16))
      {
        os<<lanes; return;
      }
      if(fail && lanes==1)
      {
        if(t.is_uint())
        {
          if (t.bits() > 64) {
            os << "uint" << "64" << "_t"; return;
          } else {
            std::string str;
            if      (t.bits() <= 8)  str = "8";
            else if (t.bits() <= 16) str = "16";
            else if (t.bits() <= 32) str = "32";
            else                   str = "64";
            os<< "uint"<<  str  <<"_t"; return;
          }
        }
        if(t.is_int())
        {
          if (t.bits() > 64) {
            os << "int" << "64" << "_t"; return;
          } else {
            std::string str;
            if      (t.bits() <= 8)  str = "8";
            else if (t.bits() <= 16) str = "16";
            else if (t.bits() <= 32) str = "32";
            else                   str = "64";
            os << "int" << str << "_t"; return;
          }
        }
      }
    }

    LOG(FATAL) << "Cannot convert type"<<t<<"to AOCL type";
  };

  void VisitStmt_(const AttrStmt* op) {
     if (op->attr_key == ir::attr::device_scope) {
      // print top( ... in host and enter fpga scope 
      if (op->value.as<StringImm>()->value == "fpga" && !fpga_scope_) {
        fpga_scope_ = true;
        PrintIndent();
        
        // track the stream usage
        var2nameType unreg_vars;
        StreamCollector collector(arg_vars, stream_table, "cpu");
        collector.Visit(op->body);
        // update data type and name 
        for (size_t k = 0; k < arg_vars.size(); k ++)
          arg_top_vars[arg_vars[k]]; 
        for (auto k : collector.host_undefined_) 
          arg_top_vars[k.get()];
        TypeCollector visitor(arg_top_vars);
        visitor.Visit(op->body);
  
        // generte function calls 
        stream << "top(";
        // int index = 0;
        // for (auto op : stream_stmts) {
        //   if (index !=0) stream << ", ";
        //   std::string vid;
        //   if (!var_idmap_.count(op->buffer_var.get())) 
        //     vid = AllocVarID(op->buffer_var.get());
        //   else vid = GetVarID(op->buffer_var.get());
        //   stream << vid;
        //   if (vid.find("stream_send") != std::string::npos || 
        //       vid.find("stream_recv") != std::string::npos) {
        //     if (index !=0) arg_stream << ", ";
        //     PrintType(op->buffer_var.type(), arg_stream);
        //     arg_stream << " " << vid;
        //   } 
        //   index++;
        // }
        // for (auto op : stream_exprs) {
        //   if (index !=0) stream << ", ";
        //   std::string vid;
        //   if (!var_idmap_.count(op->buffer_var.get())) 
        //     vid = AllocVarID(op->buffer_var.get());
        //   else vid = GetVarID(op->buffer_var.get());
        //   stream << vid;
        //   // stream << op->buffer_var.get()->name_hint;
        //   if (vid.find("stream_send") != std::string::npos || 
        //       vid.find("stream_recv") != std::string::npos) {
        //     if (index !=0) arg_stream << ", ";
        //     PrintType(op->buffer_var.type(), arg_stream);
        //     arg_stream << " " << vid;
        //   } 
        //   index++;
        // }
        stream << ");\n";
  
        // switch context to device scope
        host_stream << this->stream.str();
        this->stream.str("");
        this->stream.clear();
  
      // swtich from device to host
      } else if (op->value.as<StringImm>()->value == "cpu" && 
                 fpga_scope_) {
        fpga_scope_ = false;
        device_stream << this->stream.str();
        this->stream.str("");
        this->stream.clear();
      }
    }
    CodeGenC::VisitStmt_(op);
  }

    void VisitStmt_(const Allocate* op) {
      std::string vid = AllocVarID(op->buffer_var.get());
      if (vid.find("stream_") != std::string::npos) { 
        // do not print alloc stream 
        this->PrintStmt(op->body);
      } else {
        CHECK(!is_zero(op->condition));
        this->PrintIndent();
        int32_t constant_size = op->constant_allocation_size();
        CHECK_GT(constant_size, 0)
            << "Can only handle constant size stack allocation for now";
        const Variable* buffer = op->buffer_var.as<Variable>();
        var_shape_map_[buffer] = op->extents;
        std::string scope = alloc_storage_scope_.at(buffer);
        PrintStorageScope(scope, stream);

        // initlize hls stream channel
        if (vid.find("stream_in") != std::string::npos || 
            vid.find("stream_out") != std::string::npos) {
          stream << "hls::stream<";
          PrintType(op->type, stream);
          stream << "> " << vid << ";\n";
        } else {
          PrintType(op->type, stream);
          stream << ' '<< vid;
          if (constant_size > 1) {// Transfer length one array to scalar
            for (size_t i = 0; i < op->extents.size(); i++) {
              stream << '[';
              PrintExpr(op->extents[i], stream);
              stream << "]";
            }
          }
          stream << ";\n";
        }
        buf_length_map_[buffer] = constant_size;
        RegisterHandleType(op->buffer_var.get(), op->type);
        for (size_t i = 0; i < op->attrs.size(); i++) {
          this->PrintStmt(op->attrs[i]);
        }
        this->PrintStmt(op->body);
      }
    };

    void VisitExpr_(const StreamExpr* op, std::ostream& os) {
      std::string vid;
      if (!var_idmap_.count(op->buffer_var.get())) 
        vid = AllocVarID(op->buffer_var.get());
      else vid = GetVarID(op->buffer_var.get());
      // os << vid << ".read()";
    };

    void VisitStmt_(const Store* op) {
      std::string vid = GetVarID(op->buffer_var.get());
      if (vid.find("stream_") == std::string::npos)
        CodeGenC::VisitStmt_(op);
    };

    void VisitStmt_(const StreamStmt* op) {
      std::string vid;
      if (!var_idmap_.count(op->buffer_var.get())) 
        vid = AllocVarID(op->buffer_var.get());
      else vid = GetVarID(op->buffer_var.get());
      PrintIndent();
      auto load_op = op->value.as<Load>(); 
      auto v = load_op->buffer_var.as<Variable>();
      // placeholder args using recv name 
      if (stream_table.count(v)) {
        auto tuple = arg_top_vars[v];
        arg_top_vars[v] = std::make_tuple(vid, std::get<1>(tuple),
                                          std::get<2>(tuple));
        stream_table[v] = true;
      } // else: streamed externop defined in analysis
      // PrintExpr(op->value, stream);
      // stream << vid << ".write()\n";
    };

    void VisitStmt_(const LetStmt* op) {
      std::string value = PrintExpr(op->value);
      // Skip the argument retrieving assign statement
      std::string vid = AllocVarID(op->var.get());
      if (op->var.type() != Handle() &&
          value.find("TVMArray") == std::string::npos &&
          value.find("arg") != 0) {
        PrintIndent();
        PrintType(op->var.type(), this->stream);
        this->stream << ' '
                     << vid
                     << " = " << value << ";\n";
      // locate arg data and update arg_top_vars
      } else if (value.find("data") != std::string::npos ||
                 value.substr(0, 3) == "arg") {
        auto v = op->var.get();
        auto tuple = arg_top_vars[v]; 
        arg_vars.push_back(v);
        stream_table[v] = false;
        var_idmap_[v] = "arg_top_" + std::to_string(arg_top_count);
        arg_top_vars[v] = std::make_tuple(vid, std::get<1>(tuple),
                                          std::get<2>(tuple));
        arg_top_count += 1;
      }
      PrintStmt(op->body);
    };

};

// unified simulation function for diff platforms 
template<class CGHost, class CGXcel>
runtime::Module BuildSimModule(Array<LoweredFunc> funcs,
                               Array<Expr> attrs,
                               Array<Expr> values) {
  CodeAnalysMerlinC ca;
  CGHost cg_host;
  CGXcel cg_dev;
  
  for (LoweredFunc f : funcs) {
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    cg_host.AddFunction(f, map_arg_type);
    cg_dev.AddFunction(f, map_arg_type);
  }
  // process info: shape type and stream 
  auto& arg_vars = cg_dev.arg_vars;
  auto& stream_table = cg_dev.stream_table;
  auto& arg_top_vars = cg_dev.arg_top_vars;

  std::vector<std::tuple<bool, Type, std::vector<int>>> arg_type;
  for (size_t i = 0 ; i < arg_vars.size(); i++) {
    auto v = arg_vars[i];
    auto nameType = arg_top_vars[v];
    bool is_stream;
    if (stream_table[v])
      is_stream = true;
    else is_stream = false;
    auto item = std::make_tuple(is_stream, std::get<1>(nameType), 
                                std::get<2>(nameType));
    arg_type.push_back(item);
  }
  // tool option mapping and platform 
  std::string platform = values[0].as<StringImm>()->value;
  std::unordered_map<std::string, std::string> options;
  for (size_t k = 1; k < attrs.size(); k++) {
    auto key = attrs[k].as<StringImm>()->value;
    auto val = values[k].as<StringImm>()->value;
    options[key] = val;
  }
  return runtime::CreateSimModule(funcs[0], 
                                  cg_host.GetHost(),
                                  cg_dev.GetDevice(),
                                  arg_type, platform, options);
}

TVM_REGISTER_API("codegen.build_sim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    // dispatch to corr codegen
    auto& sptr = args[2].node_sptr();
    CHECK(sptr->is_type<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(sptr.get());
    auto data = n->data[static_cast<size_t>(0)];

    // create module node for simulation 
    std::string type = Expr(data).as<StringImm>()->value;
    if (type == "rocket") {
      *rv = BuildSimModule<CodeGenRV64PPAC, CodeGenRV64PPAC>
                (args[0], args[1], args[2]);
    } else if (type == "sdaccel") {
      *rv = BuildSimModule<CodeGenHost, CodeGenXcel>
                (args[0], args[1], args[2]);
    } else if (type == "vivado_hls") {
      *rv = BuildSimModule<CodeGenVivadoHLS, CodeGenVivadoHLS>
                (args[0], args[1], args[2]);
    } else {
    }
  });

}  // namespace codegen
}  // namespace TVM
