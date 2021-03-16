#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#undef CHECK

#include <ir/IROperator.h>
#include <base/Stencil.h>

using namespace Halide;
using namespace Halide::Internal;
using namespace std;

VarExpr a = VarExpr("a");
VarExpr b = VarExpr("b");
VarExpr c = VarExpr("c");
VarExpr d = VarExpr("d");
VarExpr x = VarExpr("x");
VarExpr y = VarExpr("y");
VarExpr z = VarExpr("z");
VarExpr t = VarExpr("t");
Expr expr_0 = Expr(0);
Expr expr_42 = Expr(42);
Expr dummy_expr = const_true();
Stmt dummy_stmt = Evaluate::make(dummy_expr);
ForType for_type = ForType::Serial;
DeviceAPI dev_api = DeviceAPI::None;

TEST_CASE("rectangular iteration domain") {
  /**
   * l2: for (z in 0 to 42) {
   *   l1: for (y in 0 to 42) {
   *     l0: for (x in 0 to 42) {
   *       true;
   *     }
   *   }
   * }
   */
  Stmt l0 = For::make(x, expr_0, expr_42, for_type, dev_api, dummy_stmt);
  Stmt l1 = For::make(y, expr_0, expr_42, for_type, dev_api, l0);
  Stmt l2 = For::make(z, expr_0, expr_42, for_type, dev_api, l1);
  REQUIRE(Stencil::GetStencil(l0));
  REQUIRE(Stencil::GetStencil(l1));
  REQUIRE(Stencil::GetStencil(l2));
}

TEST_CASE("triangular iteration domain") {
  /**
   * l2: for (z in 0 to 42) {
   *   l1: for (y in 0 to z) {
   *     l0: for (x in 0 to y+z) {
   *       true;
   *     }
   *   }
   * }
   */
  Stmt l0 = For::make(x, expr_0, y+z, for_type, dev_api, dummy_stmt);
  Stmt l1 = For::make(y, expr_0, z, for_type, dev_api, l0);
  Stmt l2 = For::make(z, expr_0, expr_42, for_type, dev_api, l1);
  REQUIRE(not Stencil::GetStencil(l0));
  REQUIRE(not Stencil::GetStencil(l1));
  REQUIRE(Stencil::GetStencil(l2));
}

TEST_CASE("dynamic loop bound") {
  /**
   * l1: for (y in 0 to 42) {
   *   l0: for (x in 0 to t) {
   *     true;
   *   }
   * }
   * l2: for (z in 0 to z) {
   *   l0: for (x in 0 to t) {
   *     true;
   *   }
   * }
   */
  Stmt l0 = For::make(x, expr_0, t, for_type, dev_api, dummy_stmt);
  Stmt l1 = For::make(y, expr_0, expr_42, for_type, dev_api, l0);
  Stmt l2 = For::make(z, expr_0, z, for_type, dev_api, l0);
  REQUIRE(not Stencil::GetStencil(l0));
  REQUIRE(not Stencil::GetStencil(l1));
  REQUIRE(not Stencil::GetStencil(l2));
}

TEST_CASE("find accessed vars") {
  /**
   * s2: a = 0;
   * s1: b = a;
   * s0: c = a*(d = b+t);
   */
  Expr e0 = Let::make(d, b+t, d);
  Stmt s0 = LetStmt::make(c, a*e0, dummy_stmt);
  Stmt s1 = LetStmt::make(b, a, s0);
  Stmt s2 = LetStmt::make(a, expr_0, s1);
  VarExprUnorderedSet vars = {b, d, t};
  REQUIRE(vars == AccessedVars::GetAccessedVars(e0));
  vars.insert(a);
  REQUIRE(vars == AccessedVars::GetAccessedVars(s0));
  REQUIRE(vars == AccessedVars::GetAccessedVars(s1));
  REQUIRE(vars == AccessedVars::GetAccessedVars(s2));
}

TEST_CASE("find local vars") {
  /**
   * s2: a = 0;
   * s1: b = a;
   * s0: c = a*(d = b+t);
   */
  Expr e0 = Let::make(d, b+t, d);
  Stmt s0 = LetStmt::make(c, a*e0, dummy_stmt);
  Stmt s1 = LetStmt::make(b, a, s0);
  Stmt s2 = LetStmt::make(a, expr_0, s1);
  VarExprUnorderedSet vars = {c, d};
  REQUIRE(vars == LocalVars::GetLocalVars(s0));
  vars.insert(b);
  REQUIRE(vars == LocalVars::GetLocalVars(s1));
  vars.insert(a);
  REQUIRE(vars == LocalVars::GetLocalVars(s2));
}

TEST_CASE("find loads") {
  /**
   * s0: d = c[x*a+b]);
   */
  Expr ld = Load::make(Int(32), c, x*a+b, dummy_expr);
  Stmt s0 = LetStmt::make(d, ld, dummy_stmt);
  ExprUnorderedSet loads = {ld};
  REQUIRE(loads == Loads::GetLoads(s0));
}

TEST_CASE("find stores") {
  /**
   * s0: c[x*a+b] = d;
   */
  Stmt s0 = Store::make(c, d, x*a+b, dummy_expr);
  unordered_set<Stmt> stores = {s0};
  REQUIRE(stores == Stores::GetStores(s0));
}

TEST_CASE("stencil load index") {
  /**
   * l1: for (y in 0 to 42) {
   *   l0: for (x in 0 to 42) {
   *     s0: d = c[x*a+b]);
   *   }
   * }
   * l3: for (y in 0 to 42) {
   *   l2: for (x in 0 to 42) {
   *     s1: d = c[x*42+42]);
   *   }
   * }
   */
  Expr e0 = Load::make(Int(32), c, x*a+b, dummy_expr);
  Expr e1 = Load::make(Int(32), c, x*expr_42+expr_42, dummy_expr);
  Stmt s0 = LetStmt::make(d, e0, dummy_stmt);
  Stmt s1 = LetStmt::make(d, e1, dummy_stmt);
  Stmt l0 = For::make(x, expr_0, expr_42, for_type, dev_api, s0);
  Stmt l1 = For::make(y, expr_0, expr_42, for_type, dev_api, l0);
  Stmt l2 = For::make(x, expr_0, expr_42, for_type, dev_api, s1);
  Stmt l3 = For::make(y, expr_0, expr_42, for_type, dev_api, l2);
  REQUIRE(not Stencil::GetStencil(l1));
  REQUIRE(Stencil::GetStencil(l3));
}

TEST_CASE("stencil store index") {
  /**
   * l1: for (y in 0 to 42) {
   *   l0: for (x in 0 to 42) {
   *     s0: c[x*a+b] = d;
   *   }
   * }
   * l3: for (y in 0 to 42) {
   *   l2: for (x in 0 to 42) {
   *     s1: c[x*42+42] = d;
   *   }
   * }
   * l5: for (y in 0 to 42) {
   *   l4: for (x in 0 to 42) {
   *     s2: c[x*42*42+42] = d;
   *   }
   * }
   */
  Stmt s0 = Store::make(c, d, x*a+b, dummy_expr);
  Stmt s1 = Store::make(c, d, x*expr_42+expr_42, dummy_expr);
  Stmt s2 = Store::make(c, d, x*expr_42*expr_42+expr_42, dummy_expr);
  Stmt l0 = For::make(x, expr_0, expr_42, for_type, dev_api, s0);
  Stmt l1 = For::make(y, expr_0, expr_42, for_type, dev_api, l0);
  Stmt l2 = For::make(x, expr_0, expr_42, for_type, dev_api, s1);
  Stmt l3 = For::make(y, expr_0, expr_42, for_type, dev_api, l2);
  Stmt l4 = For::make(x, expr_0, expr_42, for_type, dev_api, s2);
  Stmt l5 = For::make(y, expr_0, expr_42, for_type, dev_api, l4);
  REQUIRE(not Stencil::GetStencil(l1));
  REQUIRE(Stencil::GetStencil(l3));
  REQUIRE(Stencil::GetStencil(l5));
}
