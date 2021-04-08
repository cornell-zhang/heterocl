from ._ffi.function import register_func
import os, subprocess, time, re, glob
from ..report import parse_xml
from ..devices import Project, Platform
from ..autosa import autosa_infer_types

def replace_text(f_name, prev, new):
    with open(f_name, 'r') as fp:
        data = fp.read()
    data = data.replace(prev, new)
    with open(f_name, 'w') as fp:
        fp.write(data)

def indent(num):
    return " " * num

def run_process(cmd, pattern=None, env=None, debug=False):
    if debug: print("[ DEBUG ] Running commands: \n{}\n".format(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err: raise RuntimeError("Error raised: ", err.decode())
    if pattern: return re.findall(pattern, out.decode("utf-8"))
    if debug: 
        print("[ DEBUG ] Commands outputs: \n{}\n".format(out.decode("utf-8")))
    return out.decode("utf-8")

@register_func
def process_extern_module(attr_key, keys, values, code, backend):
    if attr_key == "soda":
        pos = code.find("#include")
        code = code[pos:]
        code = code.replace("extern \"C\" {", "")
        code = code.replace("}  // extern \"C\"", "")
        func_call = ""
        return [code, func_call] 

    # process the AutoSA input HLS code (string)
    elif attr_key == "autosa":
        # analyze packing and transpose information
        input_attr_info = dict()
        packed_data = list()
        transposed_data = list()
        is_axis_enabled = False
        for index in range(len(keys)):
            var = keys[index].value
            if var == "axis":
                is_axis_enabled = True
                continue
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
        if backend == "vhls":
            cmd += "--target=autosa_hls_c "
        elif backend == "aocl":
            cmd += "--target=autosa_opencl "
        else:
            raise RuntimeError(f"Illegal backend {backend}")
        cmd += "--output-dir=./autosa.tmp/output "

        # Check the env variable
        sa_space_time = os.getenv("SA_SPACE_TIME", "[3]")
        sa_array_part = os.getenv("SA_ARRAY_PAR", "[64,64,64]")
        sa_lat_hiding = os.getenv("SA_LAT_HIDING", "[16,16]")
        sa_simd = os.getenv("SA_SIMD", "[8]")
        print(f"[  INFO  ] AutoSA params: Array partition {sa_array_part}. Latency hiding {sa_lat_hiding}. SIMD{sa_simd}")

        cmd += "--sa-sizes=\"{{kernel[]->space_time{};".format(sa_space_time)
        cmd += "kernel[]->array_part{};".format(sa_array_part)
        cmd += "kernel[]->latency{};".format(sa_lat_hiding)
        cmd += "kernel[]->simd{}".format(sa_simd)
        cmd += "}\" " 
        
        # TODO: Infer reduction loops
        simd_info = os.getenv("SA_SIMD_INFO", "mm_hcl")
        cmd += "--simd-info=./autosa_tests/{}/simd_info.json ".format(simd_info)
        cmd += "--hls "
        cmd += "--hcl "
        if is_axis_enabled:
            cmd += "--axi-stream "

        # configure data packing
        if backend == "vhls":
            data_pack_config = ""
            if len(packed_data) > 0:
                data_pack_config = "--data-pack-sizes=\"{"
                delim = ""
                for var in packed_data:
                    data_pack_config += delim + "kernel[]->{}[8,32,64]".format(var) 
                    delim = ";"
                data_pack_config += "}\" "
    
            if data_pack_config == "":
                data_pack_config = "--no-data-pack "
            cmd += data_pack_config

        # addiitonal flags for intel ocl
        if backend == "aocl":
            cmd += "--loop-infinitize --double-buffer-style=0 "

        # add serialization module by default
        cmd += "--host-serialize "
        print(f"[  INFO  ] AutoSA command {cmd}")

        # dump out autosa command for debugging purposes
        with open("hcl_autosa_cmd.sh", "w") as fp:
            fp.write(cmd)
        run_process(cmd)
    
        # extract the autosa generated code
        if backend == "vhls": autosa_header = "hcl_autosa_tmp_hcl_decl.h"
        else: autosa_header = "hcl_autosa_tmp_kernel.h"

        ext = "cpp" if backend == "vhls" else "cl"
        with open(f"{autosa_dir}/autosa.tmp/output/src/hcl_autosa_tmp_kernel.{ext}", "r") as fp:
            header = fp.read() + "\n"
            header = header.replace(f"#include \"{autosa_header}\"", "")
            if backend == "aocl":
                # also extract the helper functions for data serialization and deserialization
                with open(f"{autosa_dir}/autosa.tmp/output/src/hcl_autosa_tmp_host.h", "r") as f:
                    content = f.read()
                    annotation = "/* Helper Function */"
                    start_pos = content.find(annotation)
                    end_pos = content.rfind(annotation) + len(annotation)
                    header += content[start_pos:end_pos] + "\n"

        # External module call inside top function
        with open(f"{autosa_dir}/autosa.tmp/output/src/{autosa_header}", "r") as fp:
            ret_code = fp.readlines()[0].strip() + ";\n"

        # add rules for post processing
        Project.post_proc_list["autosa.infer_types"] = autosa_infer_types

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


# Define HCL runtime behavior
@register_func
def hcl_status_control(empty, dev_hash):
    assert isinstance(Project.platform, Platform)
    p = Project.platform
    mode = Project.platform.tool.mode
    execute_arguments = p.execute_arguments

    if p.to_codegen:
        curr_time = time.strftime("%H:%M:%S", time.gmtime())
        host_file = os.path.join(Project.path, "host.cpp")
        if os.path.exists(host_file):
            print("[{}] Workdir {} exists. Skip codegen.".format(curr_time, host_file))
            return "pass"

        print("[{}] Copying harness files for platform {}...".format(curr_time, p.name))
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
        # Launch the binary and write output to JSON
        status = p.execute(Project.path, mode, **execute_arguments)
        if status:
            print("[  INFO  ] Compilation done...")
            Project.platform.execute_status = True
            return "execute"
        else:
            print("[  INFO  ] Compilation still running. Please wait...")
            return "pass"

    else: 
        print("[  INFO  ] Please consider using f.inspect/compile/execute()")
        return "pass"
