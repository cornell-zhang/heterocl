/*!
 *  Copyright (c) 2017 by Contributors
 * \file cpu_dsl_api.cc
 * \brief DSL API dispatcher
 */
#ifndef RUNTIME_DSL_API_H_
#define RUNTIME_DSL_API_H_

#include <tvm/c_dsl_api.h>

namespace TVM {
namespace runtime {
/*!
 * \brief Common interface for DSL API
 *  Used for runtime registration
 */
class DSLAPI {
 public:
  virtual void NodeFree(NodeHandle handle) const = 0;

  virtual void NodeTypeKey2Index(const char* type_key,
                                 int* out_index) const = 0;

  virtual void NodeGetTypeIndex(NodeHandle handle, int* out_index) const = 0;

  virtual void NodeGetAttr(NodeHandle handle, const char* key,
                           TVMValue* out_value, int* out_type_code,
                           int* out_success) const = 0;

  virtual void NodeListAttrNames(NodeHandle handle, int* out_size,
                                 const char*** out_array) const = 0;
};
}  // namespace runtime
}  // namespace TVM
#endif  // RUNTIME_DSL_API_H_
