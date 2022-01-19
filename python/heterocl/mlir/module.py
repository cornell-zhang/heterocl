from ..report import report_stats
from .runtime import execute_fpga_backend, execute_llvm_backend
from ..devices import Platform

class HCLModule(object):

    def __init__(self, name, src, target, context=None):
        self.name = name
        self.src = src
        self.target = target
        self.context = context

    def __call__(self, *argv):
        if "target" not in self.__dict__.keys():
            raise RuntimeError("No attached target!")
        if "name" not in self.__dict__.keys():
            raise RuntimeError("No module name specified!")
        target = self.target
        if isinstance(target, Platform) and target.tool.name == "vivado_hls":
            execute_fpga_backend(self.target)
        elif target == "llvm":
            execute_llvm_backend(self.src, self.context, self.name, *argv)
        else:
            raise RuntimeError("Not implemented")

    def report(self):
        """Get tool report
        """
        if "target" not in self.__dict__.keys():
            raise RuntimeError("No attached target!")
        if "name" not in self.__dict__.keys():
            raise RuntimeError("No module name specified!")
        target = self.target
        if target.tool.name == "vivado_hls":
            if "csyn" not in target.tool.mode:
                raise RuntimeError(
                    "Not supported mode {}. Use csyn mode to retrieve the report instead.".format(target.tool.mode))
        else:
            raise RuntimeError("Not implemented")
        return report_stats(target, target.project)
