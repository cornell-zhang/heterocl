from .code_builder import CodeBuilder
from tvm import make as _make

def if_(cond):
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of if_"
  return builders[0]._if(cond)

def else_():
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of else_"
  return builders[0]._else()

def elif_(cond):
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of elif_"
  return builders[0]._elif(cond)


def for_(begin, end, name="i", dtype="int32", for_type="serial"):
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of for_"
  return builders[0]._for(begin, end, name, dtype, for_type)

def while_(cond):
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of while_"
  return builders[0]._while(cond)

def or_(*args):
  ret = args[0]
  for i in range(1, len(args)):
    ret = _make.Or(ret, args[i])
  return ret

def and_(*args):
  ret = args[0]
  for i in range(1, len(args)):
    ret = _make.And(ret, args[i])
  return ret

def break_():
  builders = CodeBuilder.current
  assert len(builders) > 0, "Incorrect usage of break_"
  assert builders[0].in_for > 0, "Break must be used inside a for/while loop"
  builders[0].emit(_make.Break())
  builders[0].has_break = True

