from .code_builder import CodeBuilder

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
  assert len(builders) > 0, "Incorrect usage of _if"
  return builders[0]._for(begin, end, name, dtype, for_type)


