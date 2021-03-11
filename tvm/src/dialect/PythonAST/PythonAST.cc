/*!
 *  Copyright (c) 2021 by Contributors
 * \file PythonAST.cc
 */
#include <tvm/api_registry.h>
#include <dialect/PythonAST/PythonAST.h>

namespace TVM {
namespace AST {

Location LocationNode::make(std::string file_name, int line, int column) {
  std::shared_ptr<LocationNode> node = std::make_shared<LocationNode>();
  node->file_name = file_name;
  node->line = line;
  node->column = column;
  return Location(node);
}

PythonAST Var::make(Location loc, std::string name) {
  std::shared_ptr<Var> node = std::make_shared<Var>();
  node->loc = std::move(loc);
  node->name = name;
  return PythonAST(node);
}

PythonAST Add::make(Location loc, PythonAST lhs, PythonAST rhs) {
  std::shared_ptr<Add> node = std::make_shared<Add>();
  node->loc = std::move(loc);
  node->lhs = std::move(lhs);
  node->rhs = std::move(rhs);
  return PythonAST(node);
}

TVM_REGISTER_NODE_TYPE(LocationNode);
TVM_REGISTER_NODE_TYPE(Var);
TVM_REGISTER_NODE_TYPE(Add);

TVM_REGISTER_API("make.Location")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = LocationNode::make(args[0], args[1], args[2]);
    });

TVM_REGISTER_API("make.ASTVar")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Var::make(args[0], args[1]);
    });

TVM_REGISTER_API("make.ASTAdd")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Add::make(args[0], args[1], args[2]);
    });

}  // namespace AST
}  // namespace TVM
