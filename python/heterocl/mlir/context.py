from contextvars import ContextVar

ImperativeLoopNestCount = ContextVar("ImperativeLoopNestCount", default=1)
ImperativeLoopDepth = ContextVar("ImperativeLoopDepth", default=0)
StageName = ContextVar("StageName", default="")