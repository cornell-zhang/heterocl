from ._ffi.function import register_func
import os, subprocess, time, re, glob
from ..report import parse_xml
from ..devices import Project, Platform
from ..util import run_process
from ..autosa import generate_systolic_array

def replace_text(f_name, prev, new):
    with open(f_name, 'r') as fp:
        data = fp.read()
    data = data.replace(prev, new)
    with open(f_name, 'w') as fp:
        fp.write(data)

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
        return generate_systolic_array(keys, values, code, backend)

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
