# from .devices import Tool
import os, subprocess
from .devices import Tool
from . import devices

"""Define HeteroCL default tool settings"""
#pylint: disable=too-few-public-methods, too-many-return-statements

class VivadoHLS(Tool):
    def __init__(self):
        name = "vivado_hls"
        mode = "sw_sim"
        options = {
            "Frequency": "300",
            "Version":  "2019.2"
        }
        super(VivadoHLS, self).__init__(name, mode, options)

class AOCL(Tool):
    def __init__(self):
        name = "aocl"
        mode = "sw_sim"
        options = {
            "Frequency": "300",
            "Version":  "2019.2"
        }
        super(AOCL, self).__init__(name, mode, options)

class Vitis(Tool):
    def __init__(self):
        name = "vitis"
        mode = "sw_sim"
        options = {
            "Frequency": "300",
            "Version":  "2019.2"
        }
        super(Vitis, self).__init__(name, mode, options)
    
    def copy_utility(self, path, source):
        source_path = os.path.join(source, "vitis")
        command = "cp {}/* {}".format(source_path, path)
        os.system(command)
    
    def compile(self, work_path):
        pass

Tool.vivado_hls = VivadoHLS()
Tool.vitis = Vitis()
Tool.aocl = AOCL()


