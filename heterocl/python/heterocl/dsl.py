from .code_builder import CodeBuilder
from tvm import make as _make

def if_(cond):
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of _if"
  return builders[0]._if(cond)

def else_():
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of _if"
  return builders[0]._else()

def for_(begin, end, name="i", dtype="int32", for_type="serial"):
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of for_"
  return builders[0]._for(begin, end, name, dtype, for_type)

def break_():
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of break_"
  assert builders[0].in_for > 0, "Break must be used inside a for loop"
  builders[0].emit(_make.Break())
  builders[0].has_break = True

