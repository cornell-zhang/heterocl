import copy
from multiprocessing import Process

from ..devices import Platform
from ..report import report_stats
from .runtime import execute_fpga_backend, execute_llvm_backend


class HCLModule(object):

    def __init__(self, name, src, target, host_src=None, context=None, return_num=0):
        self.name = name
        self.src = src  # device src
        self.host_src = host_src
        self.target = copy.copy(target)
        self.context = context
        self.return_num = return_num

    def run_hls(self, shell=False):
        execute_fpga_backend(self.target, shell)
        report = self.report()
        report.display()

    def __call__(self, *argv):
        if "target" not in self.__dict__.keys():
            raise RuntimeError("No attached target!")
        if "name" not in self.__dict__.keys():
            raise RuntimeError("No module name specified!")
        target = self.target
        if isinstance(target, Platform) and target.tool.name in ["vivado_hls", "vitis_hls"]:
            self.run_hls(shell=True)
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


class HCLSuperModule(object):

    def __init__(self, modules):
        self.modules = modules

    def __call__(self):
        pool = []
        for module in self.modules:
            pool.append(Process(target=module.run_hls, args=(False,)))
            pool[-1].start()
        for p in pool:
            p.join()
