#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <python/pybind_utils.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/parser.h>

py::class_<StrongFunctionPtr>(m, "ScriptFunction", py::dynamic_attr())
    .def(
        "__call__",
        [](py::args args, py::kwargs kwargs) {
          HANDLE_TH_ERRORS
          auto strongPtr = py::cast<StrongFunctionPtr>(args[0]);
          Function& callee = *strongPtr.function_;
          py::object result = invokeScriptFunctionFromPython(
              callee, tuple_slice(std::move(args), 1), std::move(kwargs));
          return result;
          END_HANDLE_TH_ERRORS_PYBIND
        })
     .def(
          "save",
          [](const StrongFunctionPtr& self,
             const std::string& filename,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            Module module("__torch__.PlaceholderModule");
         module.register_attribute("training", BoolType::get(), true);
            addFunctionToModule(module, self);
            module.save(filename, _extra_files);
          },
          py::arg("filename"),
          py::arg("_extra_files") = ExtraFilesMap())

py::class_<Method>(m, "ScriptMethod", py::dynamic_attr())
    .def(
          "__call__",
          [](py::args args, py::kwargs kwargs) {
            HANDLE_TH_ERRORS
            Method& method = py::cast<Method&>(args[0]);
            return invokeScriptMethodFromPython(
                method, tuple_slice(std::move(args), 1), std::move(kwargs));
            END_HANDLE_TH_ERRORS_PYBIND
          })
    .def_property_readonly("graph", &Method::graph)
    .def_property_readonly(
        "inlined_graph",
        [](const Method& self) {
            auto g = self.function().graph()->copy();
            Inline(*g);
            return g;
        })
    .def_property_readonly(
          "schema", [](Method& m) { return m.function().getSchema(); })
    .def_property_readonly("name", &Method::name)
    .def_property_readonly(
          "code",
          [](Method& self) {
            std::vector<at::IValue> constants;
            std::vector<c10::NamedTypePtr> deps;
            PythonPrint pp(constants, deps);
            pp.printMethod(self.function());
            return pp.str();
          })
      .def_property_readonly("code_with_constants", [](Method& self) {
        std::vector<at::IValue> constants;
        std::vector<c10::NamedTypePtr> deps;
        PythonPrint pp(constants, deps);
        pp.printMethod(self.function());
        std::map<std::string, at::IValue> consts;
        int i = 0;
        for (auto const& constant : constants) {
          consts["c" + std::to_string(i)] = constant;
          i += 1;
        }
        return std::make_tuple(pp.str(), consts);
      });
  m.def(
      "_jit_script_compile",
      [](const std::string& qualname,
         const Def& def,
         ResolutionCallback rcb,
         const FunctionDefaults& defaults) {
        C10_LOG_API_USAGE_ONCE("torch.script.compile");
        const auto name = c10::QualifiedName(qualname);
        TORCH_INTERNAL_ASSERT(name.name() == def.name().name());
        return script_compile_function(name, def, defaults, std::move(rcb));
      });
  m.def("_parse_source_def", [](const std::string& src) {
    Parser p(std::make_shared<Source>(src));
    return Def(p.parseFunction(/*is_method=*/true));
  });
  m.def("parse_type_comment", [](const std::string& comment) {
    Parser p(std::make_shared<Source>(comment));
    return Decl(p.parseTypeComment());
  });
  m.def("merge_type_from_type_comment", &mergeTypesFromTypeComment);
  
static StrongFunctionPtr script_compile_function(
    const c10::QualifiedName& name,
    const Def& def,
    const FunctionDefaults& defaults,
    ResolutionCallback rcb) {
  auto cu = get_python_cu();
  auto defined_functions = cu->define(
      QualifiedName(name.prefix()),
      /*properties=*/{},
      /*propResolvers=*/{},
      {def},
      {pythonResolver(std::move(rcb))},
      nullptr,
      true);
  TORCH_INTERNAL_ASSERT(defined_functions.size() == 1);
  auto& defined = defined_functions[0];
  defined->setSchema(getSchemaWithNameAndDefaults(
      def.range(), defined->getSchema(), def.name().name(), defaults));
  StrongFunctionPtr ret(std::move(cu), defined);
  didFinishEmitFunction(ret);
  return ret;
}
