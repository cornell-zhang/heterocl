from ._ffi.function import register_func
import os, subprocess, time, re, glob
from ..report import parse_xml
from ..devices import Project, Platform
debug = True

def replace_text(f_name, prev, new):
    with open(f_name, 'r') as fp:
        data = fp.read()
    data = data.replace(prev, new)
    with open(f_name, 'w') as fp:
        fp.write(data)

def indent(num):
    return " " * num

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
def process_extern_module(attr_key, keys, values, code):
    # process the AutoSA input HLS code (string)
    if attr_key == "autosa":
        # analyze packing and transpose information
        input_attr_info = dict()
        packed_data = list()
        transposed_data = list()
        for index in range(len(keys)):
            var = keys[index].value
            try:
                is_transpose, pack_factor = values[index].value.split(",")
                input_attr_info[var] = [int(is_transpose), int(pack_factor)]
                if int(pack_factor) > 0:
                    packed_data.append(var)
                if int(is_transpose) == 1:
                    transposed_data.append(var)
            except:
                pass

        pwd = os.getcwd()
        with open("hcl_autosa_tmp.c", "w") as fp:
            fp.write("#include <stdio.h>\n")
            fp.write("int main(int argc, char **argv) {\n")
            fp.write(code)
            fp.write("}")

        header = "#include <autosa.h>\n"
        ret_code = "autosa_func(args);\n"
        autosa_dir = "/usr/src/docker_autosa"
        # autosa_dir = "/curr/jaywang/research/autosa/AutoSA"
        if not os.path.exists(autosa_dir):    
            ret_code = "// Not found AutoSA. returns function placeholder\n" + indent(6) + ret_code    
            return [header, ret_code]  

        source_path = os.path.join(pwd, "hcl_autosa_tmp.c")
        cmd = "cd {}; ".format(autosa_dir)
        cmd += "./autosa "
        cmd += "{} ".format(source_path)
        cmd += "--config=./autosa_config/autosa_config.json "
        cmd += "--target=autosa_hls_c "
        cmd += "--output-dir=./autosa.tmp/output "

        # autosa configuration
        cmd += "--sa-sizes=\"{kernel[]->space_time[3];"
        cmd += "kernel[]->array_part[16,16,16];"
        cmd += "kernel[]->latency[8,8];"

        # infer SIMD loop
        if len(transposed_data) == 0:
            cmd += "kernel[]->simd[1]"
        else:
            cmd += "kernel[]->simd[8]"

        cmd += "}\" " 
        
        cmd += "--simd-info=./autosa_tests/mm_hcl/simd_info.json "
        cmd += "--hls "
        cmd += "--hcl "

        # configure data packing
        data_pack_config = ""
        if len(packed_data) > 0:
            data_pack_config = "--data-pack-sizes=\"{"
            delim = ""
            for var in packed_data:
                data_pack_config += delim + "kernel[]->{}[8,32,64]".format(var) 
                delim = ";"
            data_pack_config += "}\""

        if data_pack_config == "":
            data_pack_config = "--no-data-pack "

        cmd += data_pack_config
        cmd += "--no-linearize-device-arrays"

        # cmd += "--host-serialize"
        run_process(cmd)

        # extract the autosa generated code
        with open(f"{autosa_dir}/autosa.tmp/output/src/hcl_autosa_tmp_kernel.cpp", "r") as fp:
            header = fp.read() + "\n"            
        with open(f"{autosa_dir}/autosa.tmp/output/src/hcl_autosa_tmp_hcl_decl.h", "r") as fp:
            ret_code = fp.readlines()[0].strip() + ";\n"

        # analyze the input code
        return [header, ret_code] 

    # process information
    assert len(keys) == len(values)
    ip_func_name = ""
    paths = []
    args_map = {}
    port_types = []

    for index in range(len(keys)):
        key = keys[index].value
        if key == "kname":
            ip_func_name = values[index].value
        elif key == "source":
            paths = values[index].value.split(":")
        elif "arg:" in key:
            tensor_name = key.replace("arg:", "")
            info = values[index].value.split(":")
            dtype = info[0]
            shape = [ int(_) for _ in info[1:] ]
            args_map[tensor_name] = [dtype, shape]
        elif key == "port_types":
            v = values[index].value.split(":")
            port_types = [ int(_) for _ in v ]
        else:
            raise RuntimeError("Unknown key {}".format(key))
    
    # Extract the kernel information
    assert len(ip_func_name) > 0
    assert len(paths) > 0
    assert len(args_map) > 0

    # Analyze the input files
    source, headers = [], []
    rproc = r"((?<=[\s:~])(\w+)\s*\(([\w\s,<>\[\].=&':/*]*?)\)\s*(const)?\s*(?={))"
    found_func_def = False
    defined_in_header = False
    extracted_args = []

    def load_txt(file_name):
        f = open(file_name)
        txt = ''.join(f.readlines())
        f.close()
        return txt

    for path in paths:
        if path.endswith(".h"):
            headers.append(path)
        elif path.endswith(".cpp") or path.endswith(".cc"):
            source.append(path)
        else:
            assert False, "Unknown input source extension {}".format(path)
        assert os.path.exists(path)

    # Search the header files
    if len(headers) > 0:
        for header in headers:
            code = load_txt(src)
            procs = [(i.group(2), i.group(3)) for i in re.finditer(rproc, code)]
            if ip_func_name in dict(procs):
                extracted_args = dict(procs)[ip_func_name].split(",")
                found_func_def = True
                defined_in_header = True
                break

    # Auto generate IP header
    else:
        for src in source:
            code = load_txt(src)
            procs = [(i.group(2), i.group(3)) for i in re.finditer(rproc, code)]
            if ip_func_name in dict(procs):
                extracted_args = dict(procs)[ip_func_name].split(",")
                found_func_def = True
                break

    # Matching the inputs and extracted args
    assert found_func_def
    extracted_args = [ _.lstrip().rstrip() for _ in extracted_args ]
    assert len(args_map) == len(extracted_args)

    # Create header automatically
    index = 0
    header_decl = "void {}(".format(ip_func_name)
    func_call_str = "{}(".format(ip_func_name)
    for k, v in args_map.items():
        dtype, shape = v
        arg_def = extracted_args[index]
        if index != 0:
            func_call_str += ", "
            header_decl += ", "

        func_call_str += k 
        header_decl += dtype + " " + k
        index += 1
    func_call_str += ");\n"
    return [header_decl, func_call_str]

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

# Define HCL runtime behavior
@register_func
def hcl_status_control(empty, dev_hash):
    assert isinstance(Project.platform, Platform)
    p = Project.platform

    if p.to_codegen:
        print("[{}] Copying harness files for platform {}...".\
            format(time.strftime("%H:%M:%S", time.gmtime()), p.name))
        path = os.path.dirname(__file__)
        path = os.path.join(path, "../harness/")
        # common harness files
        # download rapidjson to codebase if it does not exist
        rapid_json_path = os.path.join(path, "include/")
        if not os.path.exists(rapid_json_path):
            clone_cmd = "cd {}; git clone https://github.com/Tencent/rapidjson.git repo;".format(path)
            clone_cmd += "mkdir include; cp -r repo/include/rapidjson/ include/; rm -rf repo"
            run_process(clone_cmd)
        os.system("cp -r " + path + "include/* " + Project.path)
        p.copy_utility(Project.path, path)
        return "codegen"

    elif p.to_execute:
        p.execute(Project.path)
        return "execute"

# Generate harness and kernel
@register_func
def copy_and_compile(platform, mode, backend, host_only, cfg, script):
    """ Create necessary files and compile into binary """
    SUCCESS = ""

    # copy sdsoc makefile
    if platform == "sdsoc":
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
