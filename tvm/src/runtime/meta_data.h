/*!
 *  Copyright (c) 2017 by Contributors
 * \file meta_data.h
 * \brief Meta data related utilities
 */
#ifndef RUNTIME_META_DATA_H_
#define RUNTIME_META_DATA_H_

#include <dmlc/io.h>
#include <dmlc/json.h>
#include <tvm/runtime/packed_func.h>
#include <string>
#include <vector>
#include "./runtime_base.h"

namespace TVM {
namespace runtime {

/*! \brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<TVMType> arg_types;
  std::vector<std::string> thread_axis_tags;

  void Save(DMLC::JSONWriter *writer) const;
  void Load(DMLC::JSONReader *reader);
  void Save(DMLC::Stream *writer) const;
  bool Load(DMLC::Stream *reader);
};
}  // namespace runtime
}  // namespace TVM

namespace DMLC {
DMLC_DECLARE_TRAITS(has_saveload, ::TVM::runtime::FunctionInfo, true);
}  // namespace DMLC
#endif  // RUNTIME_META_DATA_H_
