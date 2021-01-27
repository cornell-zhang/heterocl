/*!
 *  Copyright (c) 2017 by Contributors
 *  Implementation of API functions related to schedule pass.
 * \file api_schedule.cc
 */
#include <tvm/api_registry.h>
#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <tvm/schedule_pass.h>
#include <tvm/tensor.h>
#include "../schedule/graph.h"

namespace TVM {
namespace schedule {

TVM_REGISTER_API("schedule.AutoInlineElemWise")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      AutoInlineElemWise(args[0]);
    });

TVM_REGISTER_API("schedule.AutoInlineInjective")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      AutoInlineInjective(args[0]);
    });

TVM_REGISTER_API("schedule.ScheduleOps")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      if (args.size() == 2)
        *ret = ScheduleOps(args[0], args[1], true);
      else
        *ret = ScheduleOps(args[0], args[1], args[2]);
    });

#define REGISTER_SCHEDULE_PASS1(PassName) \
  TVM_REGISTER_API("schedule." #PassName) \
      .set_body(                          \
          [](TVMArgs args, TVMRetValue* ret) { *ret = PassName(args[0]); })

#define REGISTER_SCHEDULE_PASS2(PassName)            \
  TVM_REGISTER_API("schedule." #PassName)            \
      .set_body([](TVMArgs args, TVMRetValue* ret) { \
        *ret = PassName(args[0], args[1]);           \
      })

REGISTER_SCHEDULE_PASS1(InferBound);
REGISTER_SCHEDULE_PASS1(ScopePartition);
REGISTER_SCHEDULE_PASS2(CreateReadGraph);
REGISTER_SCHEDULE_PASS2(PostDFSOrder);
REGISTER_SCHEDULE_PASS1(CreateAttachPath);

}  // namespace schedule
}  // namespace TVM
