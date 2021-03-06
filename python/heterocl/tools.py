# from .devices import Tool
import os, subprocess
from .devices import Tool
from . import devices
from .report import *

"""Define HeteroCL default tool settings"""
#pylint: disable=too-few-public-methods, too-many-return-statements


def run_shell_script(command):
    with open("temp.sh", "w") as fp:
        fp.write(command)
    ret = subprocess.run(command, 
        stdout=subprocess.PIPE, check=True, shell=True)
    os.remove("temp.sh")
    return ret.stdout.decode('utf-8')


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

    def copy_utility(self, path, source):
        source_path = os.path.join(source, "aocl")
        command = "cp {}/* {}".format(source_path, path)
        os.system(command)
    
    def report(self, project_name):
        res = dict()
        if self.mode == "hw_exe":
            path = os.path.join(project_name, "acl_quartus_report.txt")
            assert os.path.exists(path), path
            rpt = parse_aocl_prof_report(path)
            res["pnr"] = rpt
        return res

class Vitis(Tool):
    def __init__(self):
        name = "vitis"
        mode = "sw_sim"
        options = {
            "Frequency": "300",
            "Version":  "2019.2"
        }
        super(Vitis, self).__init__(name, mode, options)

        self.tool_mode = None
        self.xpfm = None
        self.binary = None
        self.build_dir = None

    def copy_utility(self, path, source):
        source_path = os.path.join(source, "vitis")
        command = "cp {}/* {}".format(source_path, path)
        os.system(command)
    
    def compile(self, work_path, mode, xpfm):
        if mode == "hw_exe": 
            self.tool_mode = "hw"
        elif mode == "sw_sim": 
            self.tool_mode = "sw_emu"
        elif mode == "hw_sim": 
            self.tool_mode = "hw_emu"

        self.xpfm = xpfm
        device = self.xpfm.split("/")[-1].replace(".xpfm", "")
        path = "build_dir.{}.{}".format(self.tool_mode, device)
        
        build_dir = "_x.{}.{}".format(self.tool_mode, device)
        self.build_dir = os.path.join(work_path, build_dir)

        path = os.path.join(path, "kernel.xclbin")
        binary = os.path.join(work_path, path)
        self.binary = binary 

        if not os.path.exists(self.binary):
            print("[  INFO  ] Not found {}. recompile".format(binary))
            command = "cd {}; ".format(work_path)
            command += "make all TARGET={} DEVICE={}".\
                format(self.tool_mode, xpfm)
            run_shell_script(command)
        else:
            print("[  INFO  ] Found compiled binary {}".format(binary))


    def execute(self, work_path, mode):  
        assert os.path.exists(self.binary), self.binary
        run_cmd = "cp {} {};".format(self.binary, work_path)
        run_cmd += "cd {};".format(work_path)
        if mode == "hw_exe":
            run_cmd += "./host kernel.xclbin"
        elif mode == "sw_sim":
            run_cmd += "XCL_EMULATION_MODE=sw_emu ./host kernel.xclbin"
        elif mode == "hw_sim":
            run_cmd += "XCL_EMULATION_MODE=hw_emu ./host kernel.xclbin"
        run_shell_script(run_cmd)


    def report(self):
        final = dict()
        path = os.path.join(self.build_dir, "reports/kernel/hls_reports/test_csynth.rpt")
        print("[  INFO  ] Parsing HLS report {}".format(path))

        hls_rpt = parse_vhls_report(path)
        final["hls"] = hls_rpt

        # Include runtime profiling result
        if self.tool_mode in ("hw", "hw_emu"):
            print("[  INFO  ] Parsing runtime profiling data...")
            path = self.build_dir.replace("_x.", "build_dir.")
            runtime_qor = parse_vitis_prof_report(path)
            final["runtime"] = runtime_qor

        return final

class SDAccel(Vitis):
    pass

Tool.vivado_hls = VivadoHLS()
Tool.vitis = Vitis()
Tool.aocl = AOCL()
Tool.sdaccel = SDAccel()


