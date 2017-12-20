import ast, inspect, re
import tvm, numpy
import visitor
from numbers import Number

class DFGNode():
  def __init__(self):
    self.inputs = []
    self.outputs = []
    self.ast = None

  @property
  def ast(self):
    return self.ast

  @property
  def inputs(self):
    return self.inputs

  @property
  def outputs(self):
    return self.outputs

  @ast.setter
  def ast(self, ast):
    self.ast = ast

  @ast.setter
  def inputs(self, inputs):
    self.inputs = inputps

  @ast.setter
  def outputs(self, outputs):
    self.outputs = outputs

class TVMPlaceholder(DFGNode):
  pass

class TVMVar(DFGNode):
  pass

class TVMComputeNode(DFGNode):
  pass

class For(DFGNode):
  def __init__(self):
    DFGNode.__init__(self)
    self.body = None
    self.ir = None

class IfThenElse(DFGNode):
  pass

class DataFlowGraph():
  def __init__(self, src, externs, args):
    self.src = src
    self.root = DFGNode()
    self.ast_root = ast.parse(src)
    visitor.Visitor().enter(self.ast_root, externs, args)

class MyTensor():
  def __init__(self, name, indices):
    self.name = name
    self.indices = indices


class Parser():
  def __init__(self, main_func, extern_funcs, args):
    self.main_func = main_func
    self.src = self.process_src(inspect.getsource(main_func))
    self.extern_funcs = self.process_func(extern_funcs)
    self.dfg = DataFlowGraph(self.src, self.extern_funcs, args)

  def process_src(self, src):
    #remove comments
    src = re.sub(r'#.*\n', "\n",  src)
    src = re.sub(r'\'\'\'.*\'\'\'', "\n", src, flags=re.S)
    return src

  def process_func(self, _funcs):
    funcs = {}
    for f in _funcs:
      name = f.__name__
      funcs[name] = self.process_src(inspect.getsource(f))
    return funcs

def mybuild(main_func, extern_func, args):

  p = Parser(main_func, extern_func, args)
