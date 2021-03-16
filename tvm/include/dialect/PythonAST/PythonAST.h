/*!
 *  Copyright (c) 2021 by Contributors
 * \file PythonAST.h
 */

#ifndef DIALECT_PYTHONAST_PYTHONAST_H
#define DIALECT_PYTHONAST_PYTHONAST_H

#include "../../tvm/expr.h"

namespace TVM {
namespace AST {

class Location;

/* Location for a PythonAST node */
class LocationNode : public Node {
 public:
  std::string file_name;
  int line;
  int column;

  static Location make(std::string file_name, int line, int column);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("file_name", &file_name);
    v->Visit("line", &line);
    v->Visit("column", &column);
  }

  static constexpr const char* _type_key = "Location";
  TVM_DECLARE_NODE_TYPE_INFO(LocationNode, Node);
};

class Location : public NodeRef {
 public:
  Location() {}
  explicit Location(std::shared_ptr<Node> n) : NodeRef(n) {}

  inline const LocationNode* operator->() const;
  using ContainerType = LocationNode;
};

/* The base class for all Python AST nodes */
class PythonASTNode : public Node {
 public:
  Location loc;

  static constexpr const char* _type_key = "PythonAST";
  TVM_DECLARE_BASE_NODE_INFO(PythonASTNode, Node);
};

class PythonAST : public NodeRef {
 public:
  PythonAST() {}
  explicit PythonAST(std::shared_ptr<Node> n) : NodeRef(n) {}

  inline const PythonASTNode* operator->() const;
  using ContainerType = PythonASTNode;
};

template <typename T>
class PythonASTBaseNode : public PythonASTNode {
 public:
  TVM_DECLARE_NODE_TYPE_INFO(T, PythonASTNode);
};

/* Other AST Nodes */
class Var : public PythonASTBaseNode<Var> {
 public:
   std::string name;

  static PythonAST make(Location loc, std::string name);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "PythonASTVar";
};

class Add : public PythonASTBaseNode<Add> {
 public:
  PythonAST lhs;
  PythonAST rhs;

  static PythonAST make(Location loc, PythonAST lhs, PythonAST rhs);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loc", &loc);
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
  }

  static constexpr const char* _type_key = "PythonASTAdd";
};

// implements of inline functions
inline const LocationNode* Location::operator->() const {
  return static_cast<const LocationNode*>(node_.get());
}

inline const PythonASTNode* PythonAST::operator->() const {
  return static_cast<const PythonASTNode*>(node_.get());
}

}  // namespace AST
}  // namespace TVM

#endif
