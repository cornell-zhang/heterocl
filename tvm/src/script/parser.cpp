#include <torch/csrc/jit/frontend/parser.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/csrc/jit/frontend/tree.h>
#include <torch/csrc/jit/frontend/tree_views.h>

Decl mergeTypesFromTypeComment(
    const Decl& decl,
    const Decl& type_annotation_decl,
    bool is_method) {
  auto expected_num_annotations = decl.params().size();
  if (is_method) {
    expected_num_annotations -= 1;
  }
  if (expected_num_annotations != type_annotation_decl.params().size()) {
    throw ErrorReport(decl.range())
        << "Number of type annotations ("
        << type_annotation_decl.params().size()
        << ") did not match the number of "
        << (is_method ? "method" : "function") << " parameters ("
        << expected_num_annotations << ")";
  }
  auto old = decl.params();
  auto _new = type_annotation_decl.params();
  std::vector<Param> new_params;
  size_t i = is_method ? 1 : 0;
  size_t j = 0;
  if (is_method) {
    new_params.push_back(old[0]);
  }
  for (; i < decl.params().size(); ++i, ++j) {
    new_params.emplace_back(old[i].withType(_new[j].type()));
  }
  return Decl::create(
      decl.range(),
      List<Param>::create(decl.range(), new_params),
      type_annotation_decl.return_type());
}

struct ParserImpl {
  explicit ParserImpl(const std::shared_ptr<Source>& source)
      : L(source), shared(sharedParserData()) {}

  Ident parseIdent() {
    auto t = L.expect(TK_IDENT);
    return Ident::create(t.range, t.text());
  }

  TreeRef createApply(const Expr& expr) {
    TreeList attributes;
    auto range = L.cur().range;
    TreeList inputs;
    parseArguments(inputs, attributes);
    return Apply::create(
        range,
        expr,
        List<Expr>(makeList(range, std::move(inputs))),
        List<Attribute>(makeList(range, std::move(attributes))));
  }

  static bool followsTuple(int kind) {
    switch (kind) {
      case TK_PLUS_EQ:
      case TK_MINUS_EQ:
      case TK_TIMES_EQ:
      case TK_DIV_EQ:
      case TK_MOD_EQ:
      case TK_NEWLINE:
      case '=':
      case ')':
        return true;
      default:
        return false;
    }
  }

  Expr parseExpOrExpTuple() {
    auto prefix = parseExp();
    if (L.cur().kind == ',') {
      std::vector<Expr> exprs = {prefix};
      while (L.nextIf(',')) {
        if (followsTuple(L.cur().kind))
          break;
        exprs.push_back(parseExp());
      }
      auto list = List<Expr>::create(prefix.range(), exprs);
      prefix = TupleLiteral::create(list.range(), list);
    }
    return prefix;
  }

   
