from ..report import report_stats
from .runtime import execute_fpga_backend, execute_llvm_backend
from ..devices import Platform

class HCLModule(object):

    def __init__(self, name, src, target, host_src=None, context=None, return_num=0):
        self.name = name
        self.src = src # device src
        self.host_src = host_src
        self.target = target
        self.context = context
        self.return_num = return_num

    def __call__(self, *argv):
        if "target" not in self.__dict__.keys():
            raise RuntimeError("No attached target!")
        if "name" not in self.__dict__.keys():
            raise RuntimeError("No module name specified!")
        target = self.target
        if isinstance(target, Platform) and target.tool.name == "vivado_hls":
            execute_fpga_backend(self.target)
            report = self.report()
            report.display()
        elif target == "llvm":
            execute_llvm_backend(self.src, self.name, self.return_num, *argv)
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
            if "csyn" not in target.tool.mode and target.tool.mode != "debug":
                raise RuntimeError(
                    "Not supported mode {}. Use csyn mode to retrieve the report instead.".format(target.tool.mode))
        else:
            raise RuntimeError("Not implemented")
        return report_stats(target, target.project)
