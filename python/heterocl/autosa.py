import re
import os
from .util import run_process
from .devices import Project, Platform
import copy

# Static class for entries of each SA module
class SystolicArrayRegistry(object):
    sa_module_cnt = 0

# Update HLS function names in the generated Extern IP core
def add_prefix(header, ret_code):
    # Preserved function keywords in AutoSA generated code
    function_list = [
        "autosa_func", "PE_wrapper", "PE"
    ]
    index = SystolicArrayRegistry.sa_module_cnt
    for f in function_list:
        header = header.replace(f"{f}(", f" inst{index}_{f}(")
        ret_code = ret_code.replace(f"{f}(", f" inst{index}_{f}(")
    SystolicArrayRegistry.sa_module_cnt += 1
    return header, ret_code

def infer_default_params(loop_bounds):
    assert len(loop_bounds) > 2, loop_bounds
    # TODO (Hecmay) support more generic infernece
    # Params for MatMul
    if len(loop_bounds) == 3:
        loop_bounds = [ int(_) for _ in loop_bounds ]
        m, n, k = loop_bounds
        ST = 3
        SA_dim_x = 4
        SA_dim_y = 4
        PART = f"{m},{n},{k}"
        if m > 256 or n > 256 or k > 256: LAT = "16,16"
        else: LAT = str(int(m/SA_dim_x)) + "," + str(int(n/SA_dim_y))
        SIMD = k if k <= 8 else 4
    else:
        ST = 3
        PART = "64,64,64"
        LAT = "16,16"
        SIMD = 2
    return ST, PART, LAT, SIMD

def get_ser_size(code):
    lines = code.split("\n")
    pattern = "<= (\d+);"
    size = 1
    for line in lines:
        rets = re.findall(pattern, line)
        if len(rets) > 0:
            assert len(rets) == 1
            size *= (int(rets[0])+1)
        else: continue
    return size

def host_code_buffer_resizing(host_code, tensor, new_size):
    try:
        pattern = f" {tensor}\((.*?)\)"
        size = re.findall(pattern, host_code)[0]
        host_code = host_code.replace(f" {tensor}({size})", f" {tensor}({new_size})")
        host_code = host_code.replace(f"{size}, {tensor}", f"{new_size}, {tensor}")
    except:
        pass
    if "AOCX" in host_code:
        pattern = f"_{tensor} = clCreateBuffer\(.*?, sizeof\(.*?\)\*(.*?),.*?\)"
        size = re.findall(pattern, host_code)[0]
        host_code = host_code.replace(f" {tensor}({size})", f" {tensor}({new_size})", 1)
        start_pos = host_code.find(f"{tensor} = clCreateBuffer("); assert start_pos > 0
        host_code = host_code[:start_pos] + host_code[start_pos:].replace(size, str(new_size), 1)
    return host_code

# TODO (Hecmay) AutoSA should generate helper functions in a fixed location
def extract_host_serialization(host_code, new_ret_code):
    pattern = re.compile("serialize_(.*?)\(")
    tensors = re.findall(pattern, new_ret_code)
    assert len(tensors) > 1

    annotation = "/* Helper Function */"
    start_pos = new_ret_code.find(annotation)
    end_pos = new_ret_code.rfind(annotation) + len(annotation)

    host_start_annotation = "/* HCL host function */"
    assert host_start_annotation in host_code
    intrinsics = new_ret_code[start_pos:end_pos]
    host_code = host_code.replace(host_start_annotation, intrinsics)
    new_ret_code = new_ret_code[:start_pos] + new_ret_code[end_pos:]
    for tensor in tensors:
        deser_func_name = f"host_deserialize_{tensor}"
        ser_func_name = f"host_serialize_{tensor}"

        if deser_func_name in host_code:
            start = host_code.find(deser_func_name)
            part = host_code[start:].split(annotation)[0]
            size = get_ser_size(part)
            buffer_name = f"{tensor}_dev_deser"
            host_code = host_code_buffer_resizing(host_code, buffer_name, size)
     
        elif ser_func_name in host_code:
            start = host_code.find(ser_func_name)
            part = host_code[start:].split(annotation)[0]
            size = get_ser_size(part)
            buffer_name = f"{tensor}_dev_ser"
            host_code = host_code_buffer_resizing(host_code, buffer_name, size)
    return host_code, new_ret_code

# reinterpret cast orginal pointers to target type
def autosa_infer_types(path, host_code, kernel_code):
    if "/* AutoSA post-processed infer_type */" in kernel_code:
        return host_code, kernel_code

    # Post-process AOCL code
    if kernel_code.find("OPENCL EXTENSION") > 0:
        host_code, kernel_code = extract_host_serialization(host_code, kernel_code)
        kernel_code = "/* AutoSA post-processed infer_type */\n" + kernel_code
        return host_code, kernel_code

    assert kernel_code.find("autosa_func") > 0
    index = SystolicArrayRegistry.sa_module_cnt - 1
    outer = re.compile(f"void\s+inst{index}_autosa_func\((.*?)\);")
    m = outer.search(kernel_code)
    assert m is not None, f"void inst{index}_autosa_func"
    inner_str = m.group(1)

    # find inner pairs
    target_dtype = dict()
    pairs = inner_str.split(", ")
    print(f"  - autosa. extract arg types. {pairs}")
    for pair in pairs:
        try:
            _ = pair.split(" *")
            dtype, tensor = _
            target_dtype[tensor] = dtype
        except:
            _ = pair.split(" ")
            dtype, tensor = _
            tensor = tensor.split("[")[0]
            target_dtype[tensor] = dtype            

    # replace args in autosa function call
    new_ret_code = kernel_code
    for k, v in target_dtype.items():
        new_ret_code = new_ret_code.\
            replace("buffer_{}[0]".format(k), "{}".format(k))   

    # replace the original pointer types in top function
    outer = re.compile("void test\((.*?)\)")
    m = outer.search(new_ret_code)
    inner_str = m.group(1)
    pairs = inner_str.split(", ")
    for pair in pairs:
        dtype, arg = pair.split(" ")
        arg_name = arg.split("[")[0]
        if arg_name in target_dtype:
            new_type = target_dtype[arg_name]
            new_ret_code = new_ret_code.replace(pair, f"{new_type}* {arg_name}")
    
    # TODO (Hecmay) check input buffer placement
    # this is only enabled when the input/output buffers are off-chip
    host_serialization = True
    if host_serialization:
        # Extract serilization functions from generated code
        annotation = "/* Helper Function */"
        start_pos = new_ret_code.find(annotation)
        end_pos = new_ret_code.rfind(annotation) + len(annotation)

        host_start_annotation = "/* HCL host function */"
        assert host_start_annotation in host_code
        intrinsics = new_ret_code[start_pos:end_pos]
        host_code = host_code.replace(host_start_annotation, intrinsics)
        new_ret_code = new_ret_code[:start_pos] + new_ret_code[end_pos:]

        # Serialization buffer resizing
        for tensor in target_dtype:
            deser_func_name = f"host_deserialize_{tensor}"
            ser_func_name = f"host_serialize_{tensor}"

            if deser_func_name in host_code:
                start = host_code.find(deser_func_name)
                part = host_code[start:].split(annotation)[0]
                size = get_ser_size(part)
                buffer_name = f"{tensor}_dev_deser"
                host_code = host_code_buffer_resizing(host_code, buffer_name, size)
         
            elif ser_func_name in host_code:
                start = host_code.find(ser_func_name)
                part = host_code[start:].split(annotation)[0]
                size = get_ser_size(part)
                buffer_name = f"{tensor}_dev_ser"
                host_code = host_code_buffer_resizing(host_code, buffer_name, size)

    new_ret_code = "/* AutoSA post-processed infer_type */\n" + new_ret_code 
    return host_code, new_ret_code

def generate_systolic_array(keys, values, code, backend):
    # analyze packing and transpose information
    input_attr_info = dict()
    packed_data = list()
    transposed_data = list()
    is_axis_enabled = False
    loop_bounds = list()

    # process attribute information for AutoSA module
    for index in range(len(keys)):
        key = keys[index].value
        if key == "axis":
            is_axis_enabled = True
            continue
        elif key == "loop_bound":
            loop_bounds = values[index].value.split(",")
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

    # Get the default value
    ST, PART, LAT, SIMD = infer_default_params(loop_bounds)
    # Internal debugging interface to set up the params
    sa_space_time = os.getenv("SA_SPACE_TIME", ST)
    sa_array_part = os.getenv("SA_ARRAY_PAR", PART)
    sa_lat_hiding = os.getenv("SA_LAT_HIDING", LAT)
    sa_simd = os.getenv("SA_SIMD", SIMD)

    print(f"[  INFO  ] AutoSA params: Array partition {sa_array_part}. Latency hiding {sa_lat_hiding}. SIMD{sa_simd}")
    cmd += "--sa-sizes=\"{{kernel[]->space_time[{}];".format(sa_space_time)
    cmd += "kernel[]->array_part[{}];".format(sa_array_part)
    cmd += "kernel[]->latency[{}];".format(sa_lat_hiding)
    cmd += "kernel[]->simd[{}]".format(sa_simd)
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

    header, ret_code = add_prefix(header, ret_code)
    return [header, ret_code] 