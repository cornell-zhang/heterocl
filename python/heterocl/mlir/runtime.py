import os
import re
import subprocess
from sys import platform
import time
from hcl_mlir import ir
from hcl_mlir import runtime as rt
import numpy as np
import ctypes
from ..report import parse_xml

def run_process(cmd, pattern=None, env=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err:
        raise RuntimeError("Error raised: ", err.decode())
    if pattern:
        return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")


def copy_build_files(target, script=None):
    # make the project folder and copy files
    os.makedirs(target.project, exist_ok=True)
    path = os.path.dirname(__file__)
    path = os.path.join(path, "../harness/")
    project = target.project
    platform = str(target.tool.name)
    mode = str(target.tool.mode)
    if platform == "vivado_hls":
        os.system("cp " + path + "vivado/* " + project)
        os.system("cp " + path + "harness.mk " + project)
        if mode != "custom":
            removed_mode = ["csyn", "csim", "cosim", "impl"]
            selected_mode = mode.split("|")
            for s_mode in selected_mode:
                removed_mode.remove(s_mode)

            new_tcl = ""
            with open(os.path.join(project, "run.tcl"), "r") as tcl_file:
                for line in tcl_file:
                    if (
                        ("csim_design" in line and "csim" in removed_mode)
                        or ("csynth_design" in line and "csyn" in removed_mode)
                        or ("cosim_design" in line and "cosim" in removed_mode)
                        or ("export_design" in line and "impl" in removed_mode)
                    ):
                        new_tcl += "#" + line
                    else:
                        new_tcl += line
        else:  # custom tcl
            print("Warning: custom Tcl file is used, and target mode becomes invalid.")
            new_tcl = script

        with open(os.path.join(project, "run.tcl"), "w") as tcl_file:
            tcl_file.write(new_tcl)
        return "success"
    else:
        raise RuntimeError("Not implemented")


def execute_fpga_backend(target):
    project = target.project
    platform = str(target.tool.name)
    mode = str(target.tool.mode)
    if platform == "vivado_hls":
        assert (
            os.system("which vivado_hls >> /dev/null") == 0
        ), "cannot find vivado hls on system path"
        ver = run_process("g++ --version", "\d\.\d\.\d")[0].split(".")
        assert (
            int(ver[0]) * 10 + int(ver[1]) >= 48
        ), "g++ version too old {}.{}.{}".format(ver[0], ver[1], ver[2])

        cmd = "cd {}; make ".format(project)
        if mode == "csim":
            cmd += "csim"
            out = run_process(cmd + " 2>&1")
            runtime = [k for k in out.split("\n") if "seconds" in k][0]
            print(
                "[{}] Simulation runtime {}".format(
                    time.strftime("%H:%M:%S", time.gmtime()), runtime
                )
            )

        elif "csyn" in mode or mode == "custom":
            cmd += "vivado_hls"
            print(
                "[{}] Begin synthesizing project ...".format(
                    time.strftime("%H:%M:%S", time.gmtime())
                )
            )
            subprocess.Popen(cmd, shell=True).wait()
            if mode != "custom":
                out = parse_xml(project, "Vivado HLS", print_flag=True)

        else:
            raise RuntimeError("{} does not support {} mode".format(platform, mode))
    else:
        raise RuntimeError("Not implemented")

def execute_llvm_backend(execution_engine, name, return_num, *argv):
    """
    - execution_engine: mlir.ExecutionEngine object, created in hcl.build
    - name: str, device top-level function name
    - return_num: int, the number of return values
    - argv: list-like object, a list of input and output variables
    """
    if not isinstance(argv, list):
        argv = list(argv)
    # Unwrap hcl Tensor to get numpy arrays
    argv = [arg.unwrap() for arg in argv]
    # Extract output arrays
    return_args = argv[-return_num:]
    # Convert output variables from numpy arrays to memref pointers
    return_pointers = list()
    for arg in return_args:
        memref = rt.get_ranked_memref_descriptor(arg)
        return_pointers.append(ctypes.pointer(ctypes.pointer(memref)))
    # Convert input variables from numpy arrays to memref pointers
    arg_pointers = list()
    for arg in argv[0:-return_num]:
        memref = rt.get_ranked_memref_descriptor(arg)
        arg_pointers.append(ctypes.pointer(ctypes.pointer(memref)))
    # Invoke device top-level function
    execution_engine.invoke(name, *return_pointers, *arg_pointers)
    # Copy output arrays back
    for i, return_p in enumerate(return_pointers):
        out_array = rt.ranked_memref_to_numpy(return_p[0])
        np.copyto(argv[-(len(return_args)-i)], out_array)