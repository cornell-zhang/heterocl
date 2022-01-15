from ..report import report_stats
from .runtime import execute_fpga_backend


class HCLModule(object):

    def __init__(self, name, src, target):
        self.name = name
        self.src = src
        self.target = target

    def __call__(self):
        execute_fpga_backend(self.target)

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
