from . import tensor

class Stage():

  def __init__(self, stage):
    self.stage = stage

  def compute_at(self, stage, axis):
    self.stage.compute_at(stage.stage, axis)

  def parallel(self, axis):
    self.stage.parallel(axis)

  def fuse(self, *args):
    return self.stage.fuse(*args)

  @property
  def stage(self):
    return self.stage

class Schedule():

  def __init__(self, sch):
    self.sch = sch

  def __getitem__(self, stage):
    if isinstance(stage, tensor.Tensor):
      return Stage(self.sch[stage.tensor])

  @property
  def sch(self):
    return self.sch
