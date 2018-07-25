from . import tensor

class Stage():

  def __init__(self, stage):
    self.stage = stage

  def compute_at(self, stage, axis):
    self.stage.compute_at(stage.stage, axis)

  def parallel(self, axis):
    self.stage.parallel(axis)

  def unroll(self, axis, factor=0):
    self.stage.unroll(axis, factor)

  def fuse(self, *args):
    return self.stage.fuse(*args)

  def split(self, parent, factor=None, nparts=None):
    return self.stage.split(parent, factor=factor, nparts=nparts)

  def pipeline(self, axis, initiation_interval=1):
    return self.stage.pipeline(axis, initiation_interval)

  def reorder(self, *args):
    return self.stage.reorder(*args)

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
