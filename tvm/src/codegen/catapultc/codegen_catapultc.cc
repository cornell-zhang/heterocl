/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include <tvm/build_module.h>
#include <tvm/runtime/registry.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "./codegen_catapultc.h"
#include "../merlinc/codeanalys_merlinc.h"
#include "../build_common.h"
#include "../build_soda.h"
#include "../codegen_soda.h"
#include "../../pass/stencil.h"

namespace TVM
{
  namespace codegen
  {

    struct argInfo
    {
      std::string name;
      StorageType mem_type;
      int mem_port;
      StreamType stream_type;
      int channel_depth;
      bool is_written;
    };

    void CodeGenCatapultC::AddFunction(LoweredFunc f,
                                       str2tupleMap<std::string, Type> map_arg_type)
    {
      // clear previous generated state.
      this->InitFuncState(f);
      map_arg_type_ = map_arg_type;
      // add to alloc buffer type.
      for (const auto &kv : f->handle_data_type)
        RegisterHandleType(kv.first.get(), kv.second.type());

      for (size_t i = 0; i < f->args.size(); ++i) {
        Var v = f->args[i];
        std::string vid = AllocVarID(v.get());

        // check type in the arg map
        if (map_arg_type.find(vid) == map_arg_type.end())
          LOG(WARNING) << vid << " type not found\n";
        else {
          auto arg = map_arg_type[vid];
          const BufferNode *buf = f->api_args[i].as<BufferNode>();
          if (v.type().is_handle() && buf) {
            var_shape_map_[buf->data.get()] = buf->shape;
            array_vars.insert(std::get<0>(arg));
          }
        }
      }

      // transform lower case function name to upper case
      // std::string func_name = f->name;
      // std::transform(func_name.begin(), func_name.end(), func_name.begin(), ::toupper);

      int func_scope = this->BeginScope();
      range_ = CollectIterRange(f->body);
      this->PrintStmt(f->body);
      this->EndScope(func_scope);
      // this->PrintIndent();
    }

    std::string CodeGenCatapultC::GetDevice()
    {
      return decl_stream.str() +
             module_stream.str() +
             stream.str();
    }

    std::string CodeGenCatapultC::Finish()
    {
      return decl_stream.str() +
             module_stream.str() +
             stream.str();
    }

    std::string CodeGenCatapultC::GetTop()
    {
      return head_stream.str();
    }

    void CodeGenCatapultC::PrintHeadIndent() {
      for (int i = 0; i < head_indent_; ++i) {
        this->head_stream << ' ';
      }
    }

    void CodeGenCatapultC::PrintType(Type t, std::ostream &os)
    {
      if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed())
      {
        if (t.is_uint())
          os << "ac_int<" << t.bits() << ", false>";
        else if (t.is_int())
          os << "ac_int<" << t.bits() << ", true>";
        else if (t.is_ufixed())
          os << "ap_fixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
        else
          os << "ap_fixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
      }
      else
        CodeGenC::PrintType(t, os);
    }

    void CodeGenCatapultC::VisitExpr_(const Min *op, std::ostream &os)
    { // NOLINT(*)
      os << "hls::min(";
      PrintExpr(op->a, os);
      os << ", ";
      PrintExpr(op->b, os);
      os << ")";
    }

    void CodeGenCatapultC::VisitExpr_(const Max *op, std::ostream &os)
    { // NOLINT(*)
      os << "hls::max(";
      PrintExpr(op->a, os);
      os << ", ";
      PrintExpr(op->b, os);
      os << ")";
    }

    void CodeGenCatapultC::VisitExpr_(const GetBit *op, std::ostream &os)
    {
      PrintExpr(op->a, os);
      // need to identify whether the variale is in the interface
      os << "[";
      PrintExpr(op->index, os);
      os << "]";
    }

    void CodeGenCatapultC::VisitExpr_(const GetSlice *op, std::ostream &os)
    {
      CodeGenC::PrintExpr(op->a, os);
      Expr diff = ir::Simplify(op->index_left - op->index_right);
      const int64_t *val = as_const_int(diff);
      if (val == nullptr)
        LOG(FATAL) << "The bit selection range is not a constant";
      os << ".slc<" << *val << ">(";
      PrintExpr(op->index_right, os);
      os << ")";
    }

    void CodeGenCatapultC::VisitStmt_(const Store *op)
    {
      // where does stream_vars get updated?
      std::string vid = GetVarID(op->buffer_var.get());
      // ? what if the var is not in stream_vars?
      // LOG(INFO) << "Visit store node, " << vid << "\n";
      // if (stream_vars.find(vid) != stream_vars.end())
      // {
      //   // PrintIndent();
      //   // auto bits = handle_data_type_[op->buffer_var.get()].bits();
      //   // stream << "pkt_b" << bits << " " << vid << "_temp;\n";
      //   // PrintIndent();
      //   // stream << vid << "_temp.set_data(" << PrintExpr(op->value) << ");\n";
      //   // PrintIndent();
      //   // stream << vid << "_temp.set_keep(-1);\n";
      //   // PrintIndent();
      //   // stream << vid << ".write(" << vid << "_temp);\n";
      //   return;
      // }

      // handle SetSlice
      if (const SetSlice *ss = op->value.as<SetSlice>())
      {
        Type t = op->value.type();
        Expr diff = ir::Simplify(ss->index_left - ss->index_right);
        std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
        std::string rhs = PrintExpr(ss->value);
        PrintIndent();
        stream << ref
               << ".set_slc(" << PrintExpr(ss->index_right) << ", ((ac_int<" << PrintExpr(diff)
               << ", false> const) " << rhs << "));";
      }
      else if (const SetBit *sb = op->value.as<SetBit>())
      {
        Type t = op->value.type();
        std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
        PrintIndent();
        stream << ref
               << "[" << PrintExpr(sb->index)
               << "] = " << PrintExpr(sb->value) << ";\n";
      }
      else if (auto expr_op = op->value.as<Select>())
      {
        Type t = op->value.type();
        std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
        PrintIndent();
        stream << "if (" << PrintExpr(expr_op->condition) << ") { \n";
        PrintIndent();
        stream << "  " << ref
               << " = " << PrintExpr(expr_op->true_value) << ";\n";
        PrintIndent();
        stream << "} else { \n";
        PrintIndent();
        stream << "  " << ref
               << " = " << PrintExpr(expr_op->false_value) << ";\n";
        PrintIndent();
        stream << "}\n";
      }
      else
      {
        // not falling back to CodeGenC
        CodeGenC::VisitStmt_(op);
        /*Type t = op->value.type();
        int n_elms = t.lanes();
        // LOG(INFO) << "Store else case, n_elms = " << n_elms <<"\n";
        if (n_elms == 1) {
          std::string value = this->PrintExpr(op->value);
          //std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
          this->PrintIndent();
          std::string vid = GetVarID(op->buffer_var.get());
          stream << vid << ".write( " << value << ");\n";
        } else {
          // not sure whether its correct, write more test cases
          std::string index = SSAGetID(PrintExpr(op->index), op->index.type());
          std::string value = SSAGetID(PrintExpr(op->value), op->value.type());
          std::string vid = GetVarID(op->buffer_var.get());
          for (int i = 0; i < n_elms; ++i) {
            this->PrintIndent();
            stream << vid;
            stream << ".write( ";
            PrintVecElemLoad(value, op->value.type(), i, stream);
            stream << " );\n";
          }
        }*/
      }
    }

    void CodeGenCatapultC::VisitExpr_(const Cast *op, std::ostream &os)
    { // NOLINT(*)
      // LOG(INFO) << "eliminate cast node print out\n";
      std::stringstream value;
      this->PrintExpr(op->value, value);
      os << CastFromTo(value.str(), op->value.type(), op->type);
      // os << value.str();
    }

    std::string CodeGenCatapultC::CastFromTo(std::string value, Type from, Type target)
    {
      if (from == target)
        return value;
      std::ostringstream os;
      os << "((";
      this->PrintType(target, os);
      os << ")" << value << ")";
      return os.str();
    }

    void CodeGenCatapultC::VisitExpr_(const Call *op, std::ostream &os)
    { // NOLINT(*)
      if ((op->call_type == Call::Extern ||
           op->call_type == Call::PureExtern) ||
          op->name == "sqrt")
      {
        os << "sqrt(";
        for (size_t i = 0; i < op->args.size(); i++)
        {
          this->PrintExpr(op->args[i], os);
          if (i < op->args.size() - 1)
          {
            os << ", ";
          }
        }
        os << ")";
      }
      else
      {
        CodeGenC::VisitExpr_(op, os);
      }
    }

    void CodeGenCatapultC::VisitStmt_(const Allocate *op)
    {
      CHECK(!is_zero(op->condition));
      std::string vid = AllocVarID(op->buffer_var.get());

      if (op->new_expr.defined())
      {
        CHECK_EQ(op->free_function, "nop");
        std::string new_data = PrintExpr(op->new_expr);
        this->PrintIndent();
        PrintType(op->type, stream);
        stream << "* " << vid << '=' << new_data << ";\n";
      }
      else
      {
        // LOG(INFO) << "Allocate expr not defined\n";
        int32_t constant_size = op->constant_allocation_size();
        CHECK_GT(constant_size, 0)
            << "Can only handle constant size stack allocation for now";
        const Variable *buffer = op->buffer_var.as<Variable>();
        var_shape_map_[buffer] = op->extents;

        // std::string scope; // Allocate on local scope by default
        // auto it = alloc_storage_scope_.find(buffer);
        // if (it != alloc_storage_scope_.end())
        //   scope = alloc_storage_scope_.at(buffer);
        // else
        //   scope = "local";

        // this->PrintIndent();
        PrintIndent();
        // Skip partitioned stage
        if (vid.find("_partitioned") == std::string::npos)
        {
          if (constant_size > 1)
          { 
            if (vid.find("_reuse") != std::string::npos)
            {
              PrintType(op->type, stream);
              stream << ' ' << vid;
              for (size_t i = 0; i < op->extents.size(); i++)
              {
                stream << '[';
                PrintExpr(op->extents[i], stream);
                stream << "]";
              }
            }
            else
            {
              PrintType(op->type, stream);
              stream << ' ' << vid;
              for (size_t i = 0; i < op->extents.size(); i++)
              {
                stream << '[';
                PrintExpr(op->extents[i], stream);
                stream << "]";
              }
              if (!op->init_values.empty())
              {
                stream << " = ";
                if (constant_size == 1)
                  PrintExpr(op->init_values[0], stream);
                else
                {
                  std::vector<size_t> extents;
                  for (size_t i = 0; i < op->extents.size(); i++)
                  {
                    const int64_t *extent = as_const_int(op->extents[i]);
                    CHECK(extent != nullptr) << "Extent of an init array cannot be a variable\n";
                    extents.push_back(*extent);
                  }
                  PrintArray(op->init_values, extents, stream, 0, 0);
                }
              }
            }
            stream << ";\n";
          }
          else
          {
            if (vid != "_top")
            {
              PrintType(op->type, stream);
              stream << ' ' << vid;
              stream << ";\n";
            }
            // LOG(INFO) << "allocate: constant size = 1, vid = " << vid << "\n";
          }
        }
        for (size_t i = 0; i < op->attrs.size(); i++)
          this->PrintStmt(op->attrs[i]);
        buf_length_map_[buffer] = constant_size;
      }
      // LOG(INFO) << "Visit Allocate\n";
      RegisterHandleType(op->buffer_var.get(), op->type);
      this->PrintStmt(op->body);
    }

    void CodeGenCatapultC::VisitStmt_(const For *op)
    {
      std::ostringstream os;
      if (op->for_type == ForType::Unrolled)
      {
        int unroll_factor = 0, i = 0;
        for (auto key : op->annotate_keys)
        {
          if (auto str = key.as<StringImm>())
          {
            auto factor = op->annotate_values[i].as<IntImm>();
            if (str->value == "factor" && factor != nullptr && factor->value > 1)
            {
              unroll_factor = factor->value;
              break;
            }
          }
          i++;
        }
        os << "#pragma hls_unroll";
        auto loop_bound = op->extent.as<IntImm>();
        if (unroll_factor == loop_bound->value)
          os << " yes\n";
        else if (unroll_factor > 0)
          os << " " << unroll_factor << "\n";
        else
          os << " no\n";
      }
      else if (op->for_type == ForType::Pipelined)
      {
        int II = 0, i = 0;
        for (auto key : op->annotate_keys)
        {
          if (auto str = key.as<StringImm>())
          {
            auto initiation_interval = op->annotate_values[i].as<IntImm>();
            if (str->value == "initiation_interval" &&
                initiation_interval != nullptr &&
                initiation_interval->value > 1)
            {
              II = initiation_interval->value;
              break;
            }
          }
          i++;
        }
        os << "#pragma pipeline_init_interval ";
        if (II > 0)
          os << II << "\n";
        else
          os << "\n";
      }
      GenForStmt(op, os.str(), true);
    }

    void CodeGenCatapultC::GenForStmt(const For *op, std::string pragma, bool before)
    {
      // before - whether pragma is printed before the for head statement

      std::string extent = PrintExpr(op->extent);
      std::string vid = AllocVarID(op->loop_var.get());
      CHECK(is_zero(op->min));
      if (before && pragma.length() > 0)
      {
        PrintIndent();
        stream << pragma;
      }
      PrintIndent();
      // print loop labels
      bool loop_stage_name = false;
      for (unsigned int i = 0; i < op->annotate_keys.size(); i++)
      {
        if (auto str = op->annotate_keys[i].as<StringImm>())
        {
          if (str->value == "stage_name")
          {
            loop_stage_name = true;
            auto label = op->annotate_values[i].as<StringImm>();
            std::string output_label;
            if (label->value == "")
            {
              output_label = vid;
            }
            else
            {
              output_label = label->value + "_" + vid;
            }
            for (size_t i = 0; i < output_label.size(); ++i)
            {
              if (output_label[i] == '.')
                output_label[i] = '_';
            }
            stream << output_label << ": ";
            break;
          }
        }
      }
      if (!loop_stage_name)
      {
        stream << vid << ": ";
      }
      stream << "for (";
      // PrintType(op->loop_var.type(), stream);
      // inside for statement, always use unsiged int
      stream << "unsigned";
      stream << ' ' << vid << " = 0; "
             << vid << " < " << extent
             << "; ++" << vid << ") {\n";
      if (!before && pragma.length() > 0)
      {
        PrintIndent();
        stream << pragma;
      }
      int for_scope = BeginScope();
      PrintStmt(op->body);
      this->EndScope(for_scope);
      PrintIndent();
      stream << "}\n";
    }

    void CodeGenCatapultC::VisitExpr_(const StreamExpr *op, std::ostream &os)
    {
      std::string vid = GetVarID(op->buffer_var.get());
      LOG(INFO) << "visit StreamExpr node, " << vid <<"\n";
      os << vid << ".read()";
    }

    // void CodeGenCatapultC::VisitStmt_(const ExternModule *op)
    // {
    //   PrintIndent();
    //   if (const auto *f = runtime::Registry::Get("process_extern_module"))
    //   {
    //     std::string code;
    //     code = (*f)(op->annotate_keys, op->annotate_values).operator std::string();
    //     HCL_DEBUG_LEVEL(2) << code;
    //     stream << code;
    //   }
    // }

    void CodeGenCatapultC::VisitStmt_(const StreamStmt *op)
    {
      std::string vid = GetVarID(op->buffer_var.get());
      PrintIndent();
      LOG(INFO) << "Visit StreamStmt node, " << vid << "\n";
      stream << vid << ".write(" << PrintExpr(op->value) << ");\n";
    }

    class AllocateCollector final : public IRVisitor
    {
    public:
      AllocateCollector(std::vector<const Allocate *> &alloc_list,
                        VarExprUnorderedSet &outputs)
          : alloc_list_(alloc_list), outputs_(outputs) {}

      void Visit_(const Allocate *op)
      {
        if (outputs_.count(op->buffer_var))
          alloc_list_.push_back(op);
        this->Visit(op->body);
      }

    private:
      std::vector<const Allocate *> &alloc_list_;
      VarExprUnorderedSet &outputs_;
    };

    void CodeGenCatapultC::VisitStmt_(const KernelStmt *op) {
      // print nothing
    }

    void CodeGenCatapultC::VisitStmt_(const KernelDef *op) {
      LoweredFunc f;
      // save func states
      CodeGenC::SaveFuncState(f);
      CodeGenC::InitFuncState(f);

      std::ostringstream head_decl_stream;
      std::string func_name = op->name;
      
      decl_stream << "#include <ac_int.h>\n";
      decl_stream << "#include <ac_float.h>\n";
      decl_stream << "#include <ac_channel.h>\n";
      decl_stream << "#include <mc_scverify.h>\n";
      decl_stream << "#include \"" << op->name << ".h\"\n\n";
      stream.str("");
      stream.clear();

      std::transform(func_name.begin(), func_name.end(), func_name.begin(), ::toupper);
      // assembly head file streams
      head_stream << "#ifndef " << func_name << "_H__\n";
      head_stream << "#define " << func_name << "_H__\n";
      head_stream << "#include <ac_int.h>\n"
                  << "#include <ac_channel.h>\n\n";
      head_stream << "void " << op->name << "(\n";
      // skip the first underscore
      GetUniqueName("_");
      // add to alloc buffer : type.
      for (const auto &k : op->args)
      {
        RegisterHandleType(k.get(), k.get()->type);
      }

      // collect argument information
      std::vector<argInfo> args_info;
      bool is_kernel_func = false;
      for (size_t i = 0; i < op->attributes.size(); i++)
      {
        auto info = op->attributes[i];
        CHECK(info.size() >= 2);
        auto arg_name = info[0].as<StringImm>()->value;
        for (size_t i = 0; i < arg_name.size(); ++i)
        {
          if (arg_name[i] == '.')
            arg_name[i] = '_';
        }

        if (info.size() > 2)
        {
          is_kernel_func = true;
          CHECK(info.size() == 6);
          auto mem_dev = static_cast<StorageType>(info[1].as<IntImm>()->value);
          int mem_port = info[2].as<IntImm>()->value;
          auto stream_type = static_cast<StreamType>(info[3].as<IntImm>()->value);
          int channel_depth = info[4].as<IntImm>()->value;
          bool is_written = info[5].as<IntImm>()->value == 1 ? true : false;
          argInfo arg_info = {arg_name, mem_dev, mem_port, stream_type, channel_depth, is_written};
          args_info.push_back(arg_info);
        }
        else
        {
          bool is_written = info[1].as<IntImm>()->value == 1 ? true : false;
          argInfo arg_info;
          arg_info.is_written = is_written;
          args_info.push_back(arg_info);
        }
      }

      // print top-level kernel function
      if (is_kernel_func)
      {
        stream << "#pragma design top\n";
        stream << "void CCS_BLOCK(" << op->name << ") (\n";
        for (size_t i = 0; i < op->args.size(); ++i)
        {
          VarExpr v = op->args[i];
          var_shape_map_[v.get()] = op->arg_shapes[i];
          std::string vid = AllocVarID(v.get());

          PrintIndent();
          PrintHeadIndent();
          if (i != 0) {
            stream << ", \n";
            PrintIndent();
            head_stream << ", \n";
            PrintHeadIndent();
          }
          std::string str = PrintExpr(op->arg_types[i]);
          Type type = String2Type(str);

          // pass-by-value arguments
          if (array_vars.count(vid) == 0 &&
              var_shape_map_[v.get()].size() == 1 &&
              var_shape_map_[v.get()][0].as<IntImm>()->value == 1)
          {
            PrintType(type, stream);
            PrintType(type, head_stream);
            stream << " " << vid;
            head_stream << " " << vid;

            // pass-by-pointer arguments
          }
          else
          {
            CHECK(args_info.size() > i) << i << ":" << args_info.size();
            auto info = args_info[i];

            if (info.stream_type == StreamType::FIFO)
            {
              auto bits = type.bits();
              stream << "ac_channel< "
                     << "ac_int<" << bits << ", true> > &" << vid;
              
              head_stream << "ac_channel< "
                     << "ac_int<" << bits << ", true> > &" << vid;
            }
            else // memory-mapped pointers
            {
              PrintType(type, stream);
              PrintType(type, head_stream);
              auto size = var_shape_map_[v.get()];
              stream << " " << vid;
              head_stream << " " << vid;
              for (auto &s : size)
              {
                stream << "[" << s << "]";
                head_stream << "[" << s << "]";
              }
            }
          }
        }
        stream << ") {\n";

        // No extern Keyword in Catapult
        // function body
        int func_scope = BeginScope();
        range_ = CollectIterRange(op->body);
        PrintStmt(op->body);

        EndScope(func_scope);
        PrintIndent();
        stream << "}\n\n";

        // What's the difference between top and non top kernel?
        // Non-top kernel function
      }
      else
      {
        auto const_size = [&](Array<Expr> shape) -> int32_t {
          int32_t res = 1;
          for (auto s : shape)
          {
            CHECK(s.as<IntImm>());
            auto v = s.as<IntImm>()->value;
            res = res * v;
          }
          return res;
        };
        std::ostringstream func_os;
        func_os << "static void " << op->name << "(";
        for (size_t i = 0; i < op->args.size(); ++i)
        {
          VarExpr v = op->args[i];
          var_shape_map_[v.get()] = op->arg_shapes[i];

          int32_t constant_size = const_size(op->arg_shapes[i]);
          CHECK_GT(constant_size, 0)
              << "Input arg size must be greater than 0...";
          buf_length_map_[v.get()] = constant_size;
          std::string vid = AllocVarID(v.get());
          if (i != 0)
            func_os << ", ";
          std::string str = PrintExpr(op->arg_types[i]);
          Type type = String2Type(str);

          // Scalar input
          CHECK_GT(op->arg_shapes[i].size(), 0);
          if (op->arg_shapes[i].size() == 1)
          {
            auto dim = op->arg_shapes[i][0].as<IntImm>();
            CHECK(dim);
            if (dim->value == 1 || dim->value == 0)
            {
              PrintType(type, func_os);
              auto info = args_info[i];
              if (info.is_written)
                func_os << "&";
              func_os << " " << vid;
              continue;
            }
          }

          if (op->arg_shapes[i].size() > 0)
          {
            auto shape = op->arg_shapes[i];
            PrintType(type, func_os);
            func_os << " " << vid;
            func_os << "[";
            for (size_t k = 0; k < shape.size(); k++)
            {
              if (k != shape.size() - 1)
                func_os << "][";
              func_os << shape[k];
            }
            func_os << "]";
          }
        }
        decl_stream << func_os.str() << ");\n";
        module_stream << func_os.str() << ") {\n";

        // function body
        int func_scope = BeginScope();
        range_ = CollectIterRange(op->body);
        PrintStmt(op->body);
        EndScope(func_scope);
        PrintIndent();
        module_stream << "}\n\n";
      }

      head_stream << ");\n\n" << "#endif\n";
      // LOG(INFO) << "head_stream: \n"
      //           << head_stream.str() << "\n";
      RestoreFuncState(f);
    }

  } // namespace codegen
} // namespace TVM
