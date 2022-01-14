import os
import subprocess
import time
import re
from ..report import parse_xml

class Platform(object):
    def __init__(self, platform, mode, project):
        self.platform = platform
        self.mode = mode
        self.project = project

def run_process(cmd, pattern=None, env=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err: raise RuntimeError("Error raised: ", err.decode())
    if pattern: return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")

def execute(target):
    project = target.project
    platform = target.platform
    mode = target.mode
    if platform == "vivado_hls":
        assert os.system("which vivado_hls >> /dev/null") == 0, \
            "cannot find vivado hls on system path"
        ver = run_process("g++ --version", "\d\.\d\.\d")[0].split(".")
        assert int(ver[0]) * 10 + int(ver[1]) >= 48, \
            "g++ version too old {}.{}.{}".format(ver[0], ver[1], ver[2])

        cmd = "cd {}; make ".format(project)
        if mode == "csim":
            cmd += "csim"
            out = run_process(cmd + " 2>&1")
            runtime = [k for k in out.split("\n") if "seconds" in k][0]
            print("[{}] Simulation runtime {}".format(
                time.strftime("%H:%M:%S", time.gmtime()), runtime))

        elif "csyn" in mode or mode == "custom":
            cmd += "vivado_hls"
            print("[{}] Begin synthesizing project ...".format(
                time.strftime("%H:%M:%S", time.gmtime())))
            subprocess.Popen(cmd, shell=True).wait()
            if mode != "custom":
                out = parse_xml(project, "Vivado HLS", print_flag=True)

        else:
            raise RuntimeError("{} does not support {} mode".format(platform, mode))