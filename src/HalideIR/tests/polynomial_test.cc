#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#undef CHECK

#include <arithmetic/Polynomial.h>
#include <ir/IREquality.h>
#include <ir/IROperator.h>

using namespace Halide;
using namespace Halide::Internal;
using namespace std;

VarExpr a = VarExpr("a");
VarExpr b = VarExpr("b");
VarExpr c = VarExpr("c");
VarExpr d = VarExpr("d");
VarExpr e = VarExpr("e");
VarExpr f = VarExpr("f");
VarExpr x = VarExpr("x");
VarExpr y = VarExpr("y");
VarExpr z = VarExpr("z");
VarExpr t = VarExpr("t");
Expr expr_0 = Expr(0); Expr expr_42 = Expr(42); Expr dummy_expr = const_true(); Stmt dummy_stmt = Evaluate::make(dummy_expr);
ForType for_type = ForType::Serial;
DeviceAPI dev_api = DeviceAPI::None;

void PolynomialTest(Expr func(const Expr&),
                    const Expr& expr, const Expr& ground_truth) {
  Expr transformed = func(expr);
  REQUIRE(equal(transformed, ground_truth));
  REQUIRE(graph_equal(transformed, ground_truth));
}

TEST_CASE("polynomial expansion") {
  PolynomialTest(Expand, (a+b)*c, a*c+b*c);
  PolynomialTest(Expand, (a-b)*c, a*c-b*c);
  PolynomialTest(Expand, (a+b)/c, a/c+b/c);
  PolynomialTest(Expand, (a-b)/c, a/c-b/c);
  PolynomialTest(Expand, a*(b+c), a*b+a*c);
  PolynomialTest(Expand, a*(b-c), a*b-a*c);
  PolynomialTest(Expand, a+(b+c), a+b+c);
  PolynomialTest(Expand, a+(b-c), a+b-c);
  PolynomialTest(Expand, a-(b+c), a-b-c);
  PolynomialTest(Expand, a-(b-c), a-b+c);
  PolynomialTest(
    Expand, (a+b)*(c-d)/(e+f),
    (a*c)/(e+f)-(a*d)/(e+f)+(b*c)/(e+f)-(b*d)/(e+f));
  PolynomialTest(
    Expand, (a-b)*(c+d)/(e+f),
    (a*c)/(e+f)+(a*d)/(e+f)-(b*c)/(e+f)-(b*d)/(e+f));
}

TEST_CASE("determine affine expr") {
  REQUIRE((GetAffineCoeff(d*1-b-2*a+c) == 
           VarExprInt64UnorderedMap({{a, -2}, {b, -1}, {c, 1}, {d, 1}})));
  REQUIRE((GetAffineCoeff(-d*1+b+2*a-c) ==
           VarExprInt64UnorderedMap({{a, 2}, {b, 1}, {c, -1}, {d, -1}})));
  REQUIRE((GetAffineCoeff(-c*1-b-2*a-c+50) ==
           VarExprInt64UnorderedMap({{a, -2}, {b, -1}, {c, -2},
                                     {VarExpr(), 50}})));
  REQUIRE((GetAffineCoeff(100+1-d*1-b-2*a-c+50) ==
           VarExprInt64UnorderedMap({{a, -2}, {b, -1}, {c, -1}, {d, -1},
                                     {VarExpr(), 151}})));
}
