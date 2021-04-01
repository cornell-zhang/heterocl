/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.h
 * \brief Generate Vivado HLS kernel code.
 */
#ifndef TVM_CODEGEN_CATAPULTC_H_
#define TVM_CODEGEN_CATAPULTC_H_

#include <fstream>
#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "../hlsc/codegen_hlsc.h"
#include "../merlinc/codeanalys_merlinc.h"

namespace TVM
{
  namespace codegen
  {

    class CodeGenCatapultC final : public CodeGenHLSC
    {
    public:
      void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
      std::string Finish();
      std::string GetDevice();
      std::string GetTop();
      int indent_{1};
      int head_indent_{2};

      void PrintHeadIndent();
      void PrintType(Type t, std::ostream &os) override;

      void VisitExpr_(const Min *op, std::ostream &os) override;
      void VisitExpr_(const Max *op, std::ostream &os) override;
      void VisitExpr_(const GetBit *op, std::ostream &os) override;
      void VisitExpr_(const GetSlice *op, std::ostream &os) override;
      void VisitExpr_(const StreamExpr *op, std::ostream &os) override;
      void VisitExpr_(const Call *op, std::ostream &os) override;
      void VisitExpr_(const Cast *op, std::ostream &os);
      // not overriding Load node

      void VisitStmt_(const Allocate *op) override;
      void VisitStmt_(const Store *op) override;
      void VisitStmt_(const For *op) override;
      // void VisitStmt_(const Stencil* op) override;
      void VisitStmt_(const StreamStmt *op) override;
      void VisitStmt_(const KernelDef *op) override;
      void VisitStmt_(const KernelStmt *op) override;
      // void VisitStmt_(const ExternModule *op) override;
      // not overriding Partition node

      void GenForStmt(const For *op, std::string pragma, bool before);

      virtual std::string CastFromTo(std::string value, Type from, Type target);

    private:
      std::ofstream soda_header_;
      bool sdsoc_mode{false};
      bool extern_mode{false};
      std::unordered_set<std::string> stream_vars;
      std::set<std::string> array_vars;
    };

  } // namespace codegen
} // namespace TVM

#endif // TVM_CODEGEN_CATAPULTC_H_
