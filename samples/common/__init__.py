def innermost(axis):
  # type: (heterocl.tvm.container.Array) -> heterocl.tvm.schedule.IterVar
  return axis[len(axis) - 1]

def unroll_innermost(schedule, tensor, factor=1):
  # type: (heterocl.Schedule, heterocl.Tensor, int) -> None
  schedule[tensor].unroll(innermost(tensor.axis), factor=factor)
