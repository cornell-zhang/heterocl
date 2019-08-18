/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-07-25 21:24:23
 * @LastEditTime: 2019-08-14 11:37:17
 * @LastEditors: Please set LastEditors
 */

/*
    Yang.Bai
    yb269@cornell.edu
*/

#ifndef TVM_CODEGEN_CODEGEN_SDACCEL_H_
#define TVM_CODEGEN_CODEGEN_SDACCEL_H_

# include <tvm/codegen.h>
# include <tvm/packed_func_ext.h>
# include <string>
# include "./codeanalys_openclc.h"
# include "../codegen_c.h"

namespace TVM {
namespace codegen {


class CodeGenSDACCEL final : public CodeGenC {
    public:
        CodeGenSDACCEL();
        // void AddFunction(LoweredFunc f);
        void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
        std::string Finish();

        void InitFuncState(LoweredFunc f) final;
        void BindThreadIndex(const IterVar& iv) final; // NOLINT(*)
        void PrintStorageScope(const std::string& scope, std::ostream& os) final; //NOLINT(*)
        void PrintStorageSync(const Call* op) final; //NOLINT(*)
        // void PrintType(Type t, std::ostream& os) final; //NOLINT(*)
        void PrintType(Type t, std::ostream& os) override; //NOLINT(*)

        std::string GetVecLoad(Type t, const Variable * buffer, 
                               Expr base) final; // NOLINT(*)
        void PrintVecStore(const Variable * buffer, Type t,
                           Expr base, const std::string& value) final; //NOLINT(*)
        void PrintVecAddr(const Variable * buffer, Type t,
                          Expr base, std::ostream& os); //NOLINT(*)
        std::string CastFromTo(std::string value, Type from, Type target) override; //NOLINT(*)
    
        //overload visitor
        void VisitStmt_(const LetStmt* op) final; // NOLINT(*)
        void VisitExpr_(const Broadcast * op, std::ostream& os) final; //NOLINT(*)
        void VisitExpr_(const Call * op, std::ostream& os) final; //NOLINT(*)
        void VisitExpr_(const Select * op, std::ostream& os) final; //NOLINT(*)
        void VisitExpr_(const FloatImm * op, std::ostream& os) final; //NOLINT(*)
        void VisitStmt_(const IfThenElse* op) final; //NOLINT(*)

    private:
        // whether enable fp16 and fp64 extension
        bool enable_fp16_{false};
        bool enable_fp64_{false};
   
};
} // namespace codegen
} // namespace TVM

#endif // TVM_CODEGEN_CODEGEN_SDACCEL_H_