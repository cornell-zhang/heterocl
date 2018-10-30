class _Stage():

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

    def split(self, parent, factor=None, nparts=None, mode="transform"):
        return self.stage.split(parent, factor=factor, nparts=nparts, mode=mode)

    def pipeline(self, axis, initiation_interval=1):
        return self.stage.pipeline(axis, initiation_interval)

    def reorder(self, *args):
        return self.stage.reorder(*args)

    @property
    def stage(self):
        return self.stage

class Schedule():

    stage_ops = []
    last_stages = set([])

    def __init__(self, sch):
        self.sch = sch

    def __getitem__(self, stage):
        try:
            return _Stage(self.sch[stage._op])
        except:
            return _Stage(self.sch[stage.op])

    @property
    def sch(self):
        return self.sch
