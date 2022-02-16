from contextvars import ContextVar

ImperativeLoopNestCount = ContextVar("ImperativeLoopNestCount", default=1)
ImperativeLoopDepth = ContextVar("ImperativeLoopDepth", default=0)
StageName = ContextVar("StageName", default="")


class UniqueName(object):
    scalar_idx = 0
    tensor_idx = 0
    stage_idx = 0
    schdule_idx = 0

    def __init__(self):
        pass

    @classmethod
    def get(cls, case="stage"):
        if case == "stage":
            # Imperative computing stage
            name = "stage_" + str(cls.stage_idx)
            cls.stage_idx += 1
        elif case == "scalar":
            name = "scalar_" + str(cls.scalar_idx)
            cls.scalar_idx += 1
        elif case == "tensor":
            name = "compute_" + str(cls.tensor_idx)
            cls.tensor_idx += 1
        elif case == "schedule":
            name = "schedule_" + str(cls.schedule_idx)
            cls.schedule_idx += 1
        else:
            raise RuntimeError(f"Unrecognized case in get_unique_name: {case}")
        return name
