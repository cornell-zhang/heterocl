from . import api
from ._ffi.function import register_func
import os, subprocess, time, re
from ..report import parse_xml

def replace_text(f_name, prev, new):
    with open(f_name, 'r') as fp:
        data = fp.read()
    data = data.replace(prev, new)
    with open(f_name, 'w') as fp:
        fp.write(data)

def run_process(cmd, pattern=None, env=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err: print("error raised: ", err.decode())
    if pattern: return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")

@register_func
def exec_init(dev_hash, shmids, names):
    # check whether pre-compiled bitstream exitsts
    kernel = "project/kernel.cpp"
    pre_compiled = False
    if os.path.exists(kernel):
        fp = open(kernel, "r")
        if str(dev_hash) in fp.read():
            pre_compiled = True
            print("[{}] Skip codogen. Reuse pre-generated kernel code".format(
                time.strftime("%H:%M:%S", time.gmtime())))
        fp.close()

    # check whether compiled binary exists 
    # re-compile if not. otherwise only compile host
    if pre_compiled:
        assert os.path.exists("project")
        host_file = "project/host.cpp"
        fp = open(host_file, "r")
        shmids = shmids.split("%")
        arg_names = names.split("%")
        text = fp.read()
        fp.close()

        count = 0
        for arg in arg_names:
            regex = "\/\*" + arg + "\*\/(\d+),"
            o_mid = re.findall(regex, text)[0]
            replace_text(host_file, o_mid, shmids[count])
            count = count + 1
        print("[{}] Updating host program shmids".format(
            time.strftime("%H:%M:%S", time.gmtime())))
        out = run_process("cd project; make host")

    # clean up the workspace
    else:
        if not os.path.exists("project"):
            out = run_process("mkdir -p project/save")
        out = run_process("cd project; make clean")

    return pre_compiled

@register_func
def tvm_callback_exec_evaluate(platform, mode, host_only):
    # perform simulation and extract qor
    qor = dict()

    if platform == "vivado": # to be removed?
        out = run_process("cd project; make vivado 2>&1")
        print(out)

    elif platform == "vivado_hls":

        assert os.system("which vivado_hls >> /dev/null") == 0, \
            "cannot find vivado hls on system path"
        ver = run_process("g++ --version", "\d\.\d\.\d")[0].split(".")
        assert int(ver[0]) * 10 + int(ver[1]) >= 48, \
            "g++ version too old {}.{}.{}".format(ver[0], ver[1], ver[2])

        # for host only mode
        if not os.path.isfile("project/kernel.cpp"):
            replace_text("project/Makefile", "kernel.cpp", "")
            replace_text("project/host.cpp", "#include \"kernel.h\"", "")

        cmd = "cd project; make "
        if mode == "csim":
            cmd += "csim"
            out = run_process(cmd + " 2>&1")
            runtime = [k for k in out.split("\n") if "seconds" in k][0]
            print("[{}] Simulation runtime {}".format(
                time.strftime("%H:%M:%S", time.gmtime()), runtime))

        elif "csyn" in mode:
            cmd += "vivado_hls"
            print("[{}] Begin synthesizing project ...".format(
                time.strftime("%H:%M:%S", time.gmtime())))
            out = run_process(cmd + " 2>&1")
            out = parse_xml("project", print_flag=True)

        else:
            raise RuntimeError("{} does not support {} mode".format(platform, mode))

    elif platform == "sdsoc":
        assert os.system("which sds++ >> /dev/null") == 0, \
            "cannot find sds++ on system path"
        out = run_process("cd project; make sdsoc")
        print(out)

    elif platform == "sdaccel":
        assert os.system("which xocc >> /dev/null") == 0, \
            "cannot find xocc on system path"

        if mode == "sw_sim":
            cmd = "cd project; " +\
                  "export XCL_EMULATION_MODE=sw_emu; " +\
                  "./top_function_0_host.exe -f top_function_0.sw_emu.xclbin"
            out = run_process(cmd)

        elif mode == "hw_sim":
            cmd = "cd project; " +\
                  "export XCL_EMULATION_MODE=hw_emu; " +\
                  "./top_function_0_host.exe -f top_function_0.hw_emu.xclbin"
            out = run_process(cmd)
            os.system("cat project/profile_summary.csv")

        elif mode == "hw":
            cmd = "cd project; " +\
                  "export XCL_EMULATION_MODE=hw; " +\
                  "./top_function_0_host.exe -f top_function_0.hw.xclbin"
            out = run_process(cmd)

    elif platform == "vitis":
        assert os.system("which v++ >> /dev/null") == 0, \
            "cannot find v++ on system path"
        device = os.environ["XDEVICE"].split("/")[-1]
        device = device.replace(".xpfm", "")
        cmd = "cd project; " + \
              "XCL_EMULATION_MODE=sw_emu ./host build_dir" + \
              ".sw_emu." + device + "/kernel.xclbin"
        if host_only:
            cmd = "cd project; ./host"
        out = run_process(cmd)

    elif platform == "aocl":
        cmd = "cd project; " + \
              "env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./host " + \
              " kernel.aocx"
        out = run_process(cmd)

    else:  # unsupported
        assert False, "unsupported " + platform

    return str(qor)

@register_func
def copy_and_compile(platform, mode, backend, host_only, cfg, tcl):
    """  create necessary files and compile into binary """
    path = api.__file__
    path = os.path.join(path[0:path.find("python")], "tvm/src/template/")

    if platform == "rocket":
        ppac = api.__file__ + "/hlib/rocc-ppac" 
        emulator = os.path.join(ppac, "rocket/emulator/emulator-freechips." + \
                                      "rocketchip.system-RoccExampleConfig-debug")
        # build emulator if not exist
        if not os.path.isfile(emulator):
            cmd = "cd " + ppac + ";"
            cmd += "cp src/Ppac.v rocket/src/main/resources/vsrc;" + \
                   "cp src/PpacRoCC.scala rocket/src/main/scala/tile;" + \
                   "cd rocket && git apply ../src/rocc-ppac.patch;" + \
                   "cd emulator && make CONFIG=RoccExampleConfig debug"
            # create subprocess to check
            subprocess.Popen(cmd, shell=True, stdout=open("build.log", "w")).wait()

        # re-build proxy kernel
        if not os.path.isfile(ppac + "/rocket/riscv-pk/build/pk"):
            cmd = "cd " + ppac + "/rocket/riscv-pk;"
            cmd += "git apply ../../tests/patches/riscv-pk.patch;"
            cmd += "mkdir build; cd build;"
            cmd += " ../configure --prefix=$RISCV/riscv64-unknown-elf --host=riscv64-unknown-elf;"
            cmd += "make -j8; make install"
            subprocess.Popen(cmd, shell=True, stdout=open("build.log", "w")).wait()
        return "success"

    # copy tcl and testbench  
    elif platform == "vivado_hls" or platform == "vivado":
        os.system("cp " + path + "vivado/* project/")
        os.system("cp " + path + "harness.mk project/")
        removed_mode = ["csyn","csim","cosim","impl"]
        selected_mode = mode.split("|")
        for s_mode in selected_mode:
            removed_mode.remove(s_mode)

        if tcl == "":
            new_tcl = ""
            with open("project/run.tcl","r") as tcl_file:
                for line in tcl_file:
                    if ("csim_design" in line and "csim" in removed_mode) \
                    or ("csynth_design" in line and "csyn" in removed_mode) \
                    or ("cosim_design" in line and "cosim" in removed_mode) \
                    or ("export_design" in line and "impl" in removed_mode):
                        new_tcl += "#" + line
                    else:
                        new_tcl += line
        else: # customized tcl
            print("Warning: Customized Tcl file is used, and target mode becomes invalid.")
            new_tcl = tcl

        with open("project/run.tcl","w") as tcl_file:
            tcl_file.write(new_tcl)
        return "success"

    # copy sdsoc makefile
    elif platform == "sdsoc":
        os.system("cp " + path + "sdsoc/* project/")
        os.system("cp " + path + "harness.mk project/")
        return "success"

    elif platform == "sdaccel":
        os.system("cp " + path + "sdaccel/* project/")
        os.system("cp " + path + "harness.mk project/")
        replace_text("project/Makefile", "App", "top_function_0")
        replace_text("project/utils.h", 
                     "xilinx_aws-vu9p-f1-04261818_dynamic_5_0", 
                     "xilinx_vcu1525_dynamic_5_1")
        if backend == "vhls":
          replace_text("project/Makefile", "kernel.cl", "kernel.cpp")

        # compile the program 
        assert os.system("which xocc >> /dev/null") == 0, \
            "cannot find xocc on system path"

        if mode == "sw_sim":
            env = os.environ.copy()
            assert "AWS_PLATFORM" in os.environ, \
                   "aws platform info missing" 

            # re-compile host only (reuse context ?) 
            if False and os.path.isfile("top_function_0.sw_emu.xclbin"):
              run_process("cd project; make clean; make host")
              run_process("cp top_function_0.sw_emu.xclbin project/")

            else: # config & compile
              env["XCL_EMULATION_MODE"] = "sw_emu"
              cmd = "cd project; make clean;"
              cmd += "emconfigutil --platform=$AWS_PLATFORM;"
              cmd += "make ocl OCL_TARGET=sw_emu \
                      OCL_PLATFORM=$AWS_PLATFORM \
                      APPLICATION_DIR=" + os.getcwd() + "/project/"
              out = run_process(cmd, env=env)

        # enable profiler 
        elif mode == "hw_sim":
            env = os.environ.copy()
            assert "AWS_PLATFORM" in os.environ, \
                   "aws platform info missing" 

            env["XCL_EMULATION_MODE"] = "hw_emu"
            cmd = "cd project; make clean;"
            cmd += "emconfigutil --platform=$AWS_PLATFORM;"
            cmd += "make ocl OCL_TARGET=hw_emu \
                    OCL_PLATFORM=$AWS_PLATFORM \
                    APPLICATION_DIR=" + os.getcwd() + "/project/"
            out = run_process(cmd, env=env)

        elif mode == "hw":
            env = os.environ.copy()
            assert "AWS_PLATFORM" in os.environ, \
                   "aws platform info missing" 

            env["XCL_EMULATION_MODE"] = "hw"
            cmd = "cd project; make clean;"
            cmd += "emconfigutil --platform=$AWS_PLATFORM;"
            cmd += "make ocl OCL_TARGET=hw \
                    OCL_PLATFORM=$AWS_PLATFORM \
                    APPLICATION_DIR=" + os.getcwd() + "/project/"
            out = run_process(cmd, env=env)
          
        return "success"

    elif platform == "vitis":
        env = os.environ.copy()
        assert "XDEVICE" in os.environ, \
               "vitis platform info missing" 
        os.system("cp " + path + "vitis/* project/")
        cmd = "cd project; make clean;"

        if not host_only:
            cmd += "make all TARGET=sw_emu DEVICE=$XDEVICE"
        else: cmd += "make host"
        out = run_process(cmd)
        return "success"

    elif platform == "aocl":
        env = os.environ.copy()
        assert "INTELFPGAOCLSDKROOT" in os.environ, \
               "cannot find aocl sdk for fpga on path" 

        os.system("cp " + path + "aocl/* project/")
        cmd = "cd project; make clean; make;"
        # compile kernel for xcel device
        cmd += " aoc"
        if mode == "sw_sim":
            cmd += " -march=emulator"

        cmd += " -I $INTELFPGAOCLSDKROOT/include/kernel_headers"
        cmd += " -time time.out -time-passes"
        cmd += " -v -fpc -fp-relaxed --opt-arg -nocaching"
        cmd += " -profile -report kernel.cl"
        out = run_process(cmd) 
        return "success"

    else: # unrecognized platform
        assert False, "unsupported platform " + platform
