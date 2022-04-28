/*!
 *  Copyright (c) 2017 by Contributors
 * \file ir_util.cc
 * \brief Helper functions to construct and compose IR nodes.
 */
#include "ir_util.h"

namespace TVM {
namespace ir {

Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (s.as<For>()) {
      auto n = std::make_shared<For>(*s.as<For>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<LetStmt>()) {
      auto n = std::make_shared<LetStmt>(*s.as<LetStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<AttrStmt>()) {
      auto n = std::make_shared<AttrStmt>(*s.as<AttrStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<IfThenElse>()) {
      auto n = std::make_shared<IfThenElse>(*s.as<IfThenElse>());
      CHECK(is_no_op(n->then_case));
      CHECK(!n->else_case.defined());
      n->then_case = body;
      body = Stmt(n);
    } else if (s.as<Block>()) {
      auto n = std::make_shared<Block>(*s.as<Block>());
      CHECK(is_no_op(n->rest));
      n->rest = body;
      body = Stmt(n);
    } else if (s.as<AssertStmt>()) {
      auto n = std::make_shared<AssertStmt>(*s.as<AssertStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<Allocate>()) {
      auto n = std::make_shared<Allocate>(*s.as<Allocate>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
}

Stmt MergeNest(const std::vector<std::vector<Stmt> >& nest, Stmt body) {
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}

Stmt MergeSeq(const std::vector<Stmt>& seq) {
  if (seq.size() == 0) return Evaluate::make(0);
  Stmt body = seq[0];
  for (size_t i = 1; i < seq.size(); ++i) {
    body = Block::make(body, seq[i]);
  }
  return body;
}

bool PortDirectionFinder::isOffChip(std::string name) {
  const std::string targets[1] = {"DRAM"};
    bool offchip = false;
    for (const std::string target : targets) {
      std::string name_upper = name;
      std::transform(name_upper.begin(), name_upper.end(), name_upper.begin(),
                     ::toupper);
      offchip = name_upper.find(target) != std::string::npos;
    }
    return offchip;
}

PortType PortDirectionFinder::get_direction(std::string var_name) {
    auto it_in = std::find(_in_ports.begin(), _in_ports.end(), var_name);
    auto it_out = std::find(_out_ports.begin(), _out_ports.end(), var_name);
    auto it_scl = std::find(_scalars.begin(), _scalars.end(), var_name);
    bool is_scalar = it_scl != _scalars.end();
    bool is_in = it_in != _in_ports.end();
    bool is_out = it_out != _out_ports.end();
    if (is_in && is_out) {
      bool offchip = isOffChip(var_name);
      return offchip ? PortType::OffChipMemory : PortType::Memory;
    } else if (is_in) {
      return PortType::ChannelIn;
    } else if (is_out) {
      return PortType::ChannelOut;
    } else if (is_scalar) {
      return PortType::ChannelIn;
    } else {
      LOG(FATAL) << "[SystemC Backend][PortDirectionInfer]"
                 << " cannot infer the port type and direction for port: "
                 << var_name;
    }
    return PortType::Default;
  }


}  // namespace ir
}  // namespace TVM
