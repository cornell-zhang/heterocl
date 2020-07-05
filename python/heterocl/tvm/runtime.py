from ._ffi.function import register_func
import os, subprocess, time, re, glob
from ..report import parse_xml
from ..devices import Project
debug = True

def replace_text(f_name, prev, new):
    with open(f_name, 'r') as fp:
        data = fp.read()
    data = data.replace(prev, new)
    with open(f_name, 'w') as fp:
        fp.write(data)

def run_process(cmd, pattern=None, env=None):
    if debug: print("[DEBUG] Running commands: \n{}\n".format(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err: raise RuntimeError("Error raised: ", err.decode())
    if pattern: return re.findall(pattern, out.decode("utf-8"))
    if debug: 
        print("[DEBUG] Commands outputs: \n{}\n".format(out.decode("utf-8")))
    return out.decode("utf-8")

@register_func
def exec_init(dev_hash, tool, mode):
    # check whether pre-compiled bitstream exitsts
    kernel = os.path.join(Project.path,"kernel.cpp")
    pre_compiled = False

    # check the cache 
    if mode == "hw_exe": mode = "hw"
    elif mode == "sw_sim": mode = "sw_emu"
    elif mode == "hw_sim": mode = "hw_emu"
    cache = glob.glob(os.path.join(Project.path,"save/*.xclbin"))
    target = os.path.join(Project.path,"save/{}-{}.xclbin".format(mode, dev_hash))
    if target in cache:
        pre_compiled = True
        print("[{}] Skip codogen. Found pre-built in cache".format(
            time.strftime("%H:%M:%S", time.gmtime())))
        cmd = "cp -f {} ".format(target) + os.path.join(Project.path,"kernel.xclbin")
        run_process(cmd)

    # check whether compiled binary exists 
    # re-compile if not. otherwise only compile host
    if pre_compiled:
        out = run_process("cd {}; make host".format(Project.path))

    # clean up the workspace
    else:
        if not os.path.exists(os.path.join(Project.path,"save")):
            out = run_process("mkdir -p " + os.path.join(Project.path,"save"))
        out = run_process("cd {}; make clean".format(Project.path))

    return pre_compiled

@register_func
def tvm_callback_exec_evaluate(platform, mode, host_only):
    # perform simulation and extract qor
    qor = dict()

    if platform == "vivado_hls":

        assert os.system("which vivado_hls >> /dev/null") == 0, \
            "cannot find vivado hls on system path"
        ver = run_process("g++ --version", "\d\.\d\.\d")[0].split(".")
        assert int(ver[0]) * 10 + int(ver[1]) >= 48, \
            "g++ version too old {}.{}.{}".format(ver[0], ver[1], ver[2])

        # for host only mode
        if not os.path.isfile(os.path.join(Project.path,"kernel.cpp")):
            replace_text(os.path.join(Project.path,"Makefile"), "kernel.cpp", "")
            replace_text(os.path.join(Project.path,"host.cpp"), "#include \"kernel.h\"", "")

        cmd = "cd {}; make ".format(Project.path)
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
                out = parse_xml(Project.path, print_flag=True)

        else:
            raise RuntimeError("{} does not support {} mode".format(platform, mode))

    elif platform == "sdsoc":
        assert os.system("which sds++ >> /dev/null") == 0, \
            "cannot find sds++ on system path"
        out = run_process("cd {}; make sdsoc".format(Project.path))

    elif platform == "sdaccel":
        assert os.system("which xocc >> /dev/null") == 0, \
            "cannot find xocc on system path"

        if mode == "sw_sim":
            cmd = "cd {}; ".format(Project.path) +\
                  "export XCL_EMULATION_MODE=sw_emu; " +\
                  "./top_function_0_host.exe -f top_function_0.sw_emu.xclbin"
            out = run_process(cmd)

        elif mode == "hw_sim":
            cmd = "cd {}; ".format(Project.path) +\
                  "export XCL_EMULATION_MODE=hw_emu; " +\
                  "./top_function_0_host.exe -f top_function_0.hw_emu.xclbin"
            out = run_process(cmd)
            os.system("cat " + os.path.join(Project.path,"profile_summary.csv"))

        elif mode == "hw_exe":
            cmd = "cd {}; ".format(Project.path) +\
                  "export XCL_EMULATION_MODE=hw; " +\
                  "./top_function_0_host.exe -f top_function_0.hw.xclbin"
            out = run_process(cmd)

    elif platform == "vitis":
        assert os.system("which v++ >> /dev/null") == 0, \
            "cannot find v++ on system path"
        cmd = "cd {}; ".format(Project.path)

        if mode == "hw_exe":
            cmd += "./host kernel.xclbin"
        elif mode == "sw_sim":
            cmd += "XCL_EMULATION_MODE=sw_emu ./host kernel.xclbin"
        elif mode == "hw_sim":
            cmd += "XCL_EMULATION_MODE=hw_emu ./host kernel.xclbin"

        if host_only:
            cmd = "cd {}; ./host".format(Project.path)
        out = run_process(cmd)

    elif platform == "aocl":
        if mode == "sw_sim":
            cmd = "cd {}; ".format(Project.path) + \
                  "env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./host " + \
                  " kernel.aocx"
            out = run_process(cmd)
        elif mode == "hw_sim":
            cmd = "cd {}; ".format(Project.path) + \
                  "env CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./host"
            out = run_process(cmd)

    else:  # unsupported
        assert False, "unsupported " + platform

    return str(qor)

@register_func
def copy_and_compile(platform, mode, backend, host_only, cfg, script):
    """  create necessary files and compile into binary """
    path = os.path.dirname(__file__)
    path = os.path.join(path, "../harness/")

    if platform == "rocket":
        ppac = path + "/hlib/rocc-ppac" 
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
    elif platform == "vivado_hls":
        os.system("cp " + path + "vivado/* " + Project.path)
        os.system("cp " + path + "harness.mk " + Project.path)
        if mode != "custom":
            removed_mode = ["csyn","csim","cosim","impl"]
            selected_mode = mode.split("|")
            for s_mode in selected_mode:
                removed_mode.remove(s_mode)

            new_tcl = ""
            with open(os.path.join(Project.path,"run.tcl"),"r") as tcl_file:
                for line in tcl_file:
                    if ("csim_design" in line and "csim" in removed_mode) \
                    or ("csynth_design" in line and "csyn" in removed_mode) \
                    or ("cosim_design" in line and "cosim" in removed_mode) \
                    or ("export_design" in line and "impl" in removed_mode):
                        new_tcl += "#" + line
                    else:
                        new_tcl += line
        else: # custom tcl
            print("Warning: custom Tcl file is used, and target mode becomes invalid.")
            new_tcl = script

        with open(os.path.join(Project.path,"run.tcl"),"w") as tcl_file:
            tcl_file.write(new_tcl)
        return "success"

    # copy sdsoc makefile
    elif platform == "sdsoc":
        os.system("cp " + path + "sdsoc/* " + Project.path)
        os.system("cp " + path + "harness.mk " + Project.path)
        return "success"

    elif platform == "sdaccel":
        os.system("cp " + path + "sdaccel/* " + Project.path)
        os.system("cp " + path + "harness.mk " + Project.path)
        replace_text(os.path.join(Project.path,"Makefile"), "App", "top_function_0")
        replace_text(os.path.join(Project.path,"utils.h"), 
                     "xilinx_aws-vu9p-f1-04261818_dynamic_5_0", 
                     "xilinx_vcu1525_dynamic_5_1")
        if backend == "vhls":
          replace_text(os.path.join(Project.path,"Makefile"), "kernel.cl", "kernel.cpp")

        # compile the program 
        assert os.system("which xocc >> /dev/null") == 0, \
            "cannot find xocc on system path"

        if mode == "sw_sim":
            env = os.environ.copy()
            assert "AWS_PLATFORM" in os.environ, \
                   "aws platform info missing" 

            # re-compile host only (reuse context ?) 
            if False and os.path.isfile("top_function_0.sw_emu.xclbin"):
              run_process("cd {}; make clean; make host".format(Project.path))
              run_process("cp top_function_0.sw_emu.xclbin " + Project.path)

            else: # config & compile
              env["XCL_EMULATION_MODE"] = "sw_emu"
              cmd = "cd {}; make clean;".format(Project.path)
              cmd += "emconfigutil --platform=$AWS_PLATFORM;"
              cmd += "make ocl OCL_TARGET=sw_emu \
                      OCL_PLATFORM=$AWS_PLATFORM \
                      APPLICATION_DIR=" + os.path.join(os.getcwd(),Project.path)
              out = run_process(cmd, env=env)

        # enable profiler 
        elif mode == "hw_sim":
            env = os.environ.copy()
            assert "AWS_PLATFORM" in os.environ, \
                   "aws platform info missing" 

            env["XCL_EMULATION_MODE"] = "hw_emu"
            cmd = "cd {}; make clean;".format(Project.path)
            cmd += "emconfigutil --platform=$AWS_PLATFORM;"
            cmd += "make ocl OCL_TARGET=hw_emu \
                    OCL_PLATFORM=$AWS_PLATFORM \
                    APPLICATION_DIR=" + os.path.join(os.getcwd(),Project.path)
            out = run_process(cmd, env=env)

        elif mode == "hw":
            env = os.environ.copy()
            assert "AWS_PLATFORM" in os.environ, \
                   "aws platform info missing" 

            env["XCL_EMULATION_MODE"] = "hw"
            cmd = "cd {}; make clean;".format(Project.path)
            cmd += "emconfigutil --platform=$AWS_PLATFORM;"
            cmd += "make ocl OCL_TARGET=hw \
                    OCL_PLATFORM=$AWS_PLATFORM \
                    APPLICATION_DIR=" + os.path.join(os.getcwd(),Project.path)
            out = run_process(cmd, env=env)
          
        return "success"

    elif platform == "vitis":
        env = os.environ.copy()
        assert "XDEVICE" in os.environ, \
               "vitis platform info missing" 
        os.system("cp " + path + "vitis/* " + Project.path)
        cmd = "cd {}; make clean; ".format(Project.path)

        if mode == "hw_exe": mode = "hw"
        elif mode == "sw_sim": mode = "sw_emu"
        elif mode == "hw_sim": mode = "hw_emu"

        # create connecivity config 
        with open(os.path.join(Project.path,"config.ini"), "w") as fp:
            fp.write(cfg)

        if not host_only:
            cmd += "make all TARGET=" + mode + " DEVICE=$XDEVICE"
        else: cmd += "make host"
        out = run_process(cmd)

        # mv combined binary to root and save
        device = os.environ["XDEVICE"].split("/")[-1]
        device = device.replace(".xpfm", "")
        path = os.path.join(Project.path, "build_dir.{}.{}/kernel.xclbin".format(mode, device))
        assert os.path.exists(path), "Not found {}".format(path)
        run_process("cp {} ".format(path) + os.path.join(Project.path, "kernel.xclbin"))

        kernel = os.path.join(Project.path,"kernel.cpp")
        with open(kernel, "r") as fp:
            regex = "HASH:(\d+)\n"
            hash_v = re.findall(regex, fp.read())[0]

        cache = os.path.join(Project.path,"save/{}-{}.xclbin".format(mode, hash_v))
        run_process("cp " + os.path.join(Project.path, "kernel.xclbin") + " {}".format(cache))
        return "success"

    elif platform == "aocl":
        env = os.environ.copy()

        # check aoc version 
        assert os.system("which aoc >> /dev/null") == 0, \
            "cannot find aoc on system path"
        ver = run_process("aoc --version", "\d+\.\d\.\d")[0].split(".")

        assert "INTELFPGAOCLSDKROOT" in os.environ, \
               "cannot find aocl sdk for fpga on path" 

        os.system("cp " + path + "aocl/* " + Project.path)
        cmd = "cd {}; make clean; make;".format(Project.path)

        # compile kernel for xcel device
        cmd += " aoc"
        if mode == "sw_sim":
            cmd += " -march=emulator"
        elif mode == "hw_sim":
            if int(ver[0]) < 19:
                raise RuntimeError("AOC version {}.{}.{} is too old, ".format(*ver) + \
                        "does not support hw simulation")
            cmd += " -march=simulator"

        # custom makefile flags 
        if cfg != "":
            deps = re.findall(r"deps: {(.+?)}", cfg)[0]
            custom_cmds = re.findall(r"cmds: {(.+?)}", cfg)[0]
            mk = re.findall(r"makefiles: {(.+?)}", cfg)[0]

            # copy dependency files
            out = run_process("cp -r " + deps + " " + Project.path) 
            print("[{}] Running custom commands: {}".format(
                time.strftime("%H:%M:%S", time.gmtime()), custom_cmds))
            out = run_process("cd {}; ".format(Project.path) + custom_cmds) 
            cmd += " " + mk + " "

        cmd += " -I $INTELFPGAOCLSDKROOT/include/kernel_headers"
        cmd += " -time time.out -time-passes"
        cmd += " -v -fpc -fp-relaxed -opt-arg -nocaching"
        cmd += " -profile -report kernel.cl"

        out = run_process(cmd) 
        return "success"

    else: # unrecognized platform
        assert False, "unsupported platform " + platform
