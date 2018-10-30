from .code_builder import Stage
from tvm import make as _make

def if_(cond):
  assert Stage.get_len() > 0, "Incorrect usage of if_"
  return Stage.get_cb()._if(cond)

def else_():
  assert Stage.get_len() > 0, "Incorrect usage of else_"
  return Stage.get_cb()._else()

def elif_(cond):
  assert Stage.get_len() > 0, "Incorrect usage of elif_"
  return Stage.get_cb()._elif(cond)

def for_(begin, end, step=1, name="i", dtype="int32", for_type="serial"):
  assert Stage.get_len() > 0, "Incorrect usage of for_"
  return Stage.get_cb()._for(begin, end, step, name, dtype, for_type)

def while_(cond):
  assert Stage.get_len() > 0, "Incorrect usage of while_"
  return Stage.get_cb()._while(cond)

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
  assert Stage.get_len() > 0, "Incorrect usage of break_"
  assert Stage.get_cb().for_level > 0, "Break must be used inside a for/while loop"
  Stage.get_cb().emit(_make.Break())
  Stage.get_cb().has_break = True

def return_(val):
  builders = Stage.current
  assert Stage.get_len() > 0, "Incorrect usage of return_"
  Stage.get_cb().emit(_make.Return(val))

