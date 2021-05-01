import re
import os
import copy
import sys
from .util import run_process
from .devices import Project, Platform

# Static class for entries of each SA module
class SystolicArrayRegistry(object):
    sa_module_cnt = 0

def count_SA_size(code):
    pos = code.rfind("PE_wrapper")
    function = code[pos:pos+100]
    dims = re.findall(" (\d+),", function)
    if len(dims) < 2:
        print("Failed to generate 2d SA. Size", dims)
        sys.exit()

    dimX, dimY = int(dims[0])+1, int(dims[1])+1
    print(f"[  INFO  ] generating SA dimnesion {dimX}x{dimY}.")

def indent(num):
    return " " * num

def get_function_code(name, code):
    pos = code.find(name)
    start_pos = pos - len("inline void")
    end_pos = code.find("/* Helper", pos)
    return code[start_pos:end_pos]


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

def insert_data_pack(ret_code, header, off_chip_data, written_data):
    ret_code = ret_code.replace("buffer_", "").replace("[0]", "")
    # Extract the designated data types
    pattern = re.findall("autosa_func\((.*?)\)", ret_code)[0]
    args = pattern.split(", ")
    signature = re.findall("autosa_func\((.*?)\);", header)

    # If the arg is accessed from off-chip memory, then we repalce the typedef 
    # with target packed data type
    types = signature[0].split(", ")
    for t in types:
        for arg in args:
            if arg in t:
                pattern = "_t(\d+) "
                target_type = re.findall(pattern, t)[0]
                target_type_bits = int(target_type) * 32
                # Off-chip coalesced data access
                if arg in off_chip_data:
                    header = f"#undef {arg}_t\n#define {arg}_t ap_uint<{target_type_bits}>\n" + header
                
                # Insert data packing and (de)serialization
                # Create a new buffer and reshape it to original buffer after or before AutoSA func call
                else:
                    if arg in written_data:
                        print(f"[ INFO ] Writing to on-chip memory {arg}. Packed into ap_uint<{target_type_bits}>...")
                        # ALlocate new buffer and perform data deserialization
                        deser_func = f"host_deserialize_{arg}"
                        # Check if the size matches
                        code = get_function_code(deser_func, header)
                        size = get_ser_size(code)
                        ret_code = ret_code.replace(arg, f"{arg}_sa")
                        ret_code = f"float {arg}_sa[{size}];\n" + indent(5) + ret_code + \
                            indent(6) + f"{deser_func}({arg}, {arg}_sa);\n"
                    else:
                        pass

    return ret_code, header

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
    assert len(loop_bounds) > 1, loop_bounds
    extra_flags = "--simd-info=./autosa_tests/mm_hcl/simd_info.json "
    # Params for MatMul
    if len(loop_bounds) == 3:
        loop_bounds = [ int(_) for _ in loop_bounds ]
        m, n, k = loop_bounds
        if m > 1 and n > 1 and k > 1:
            ST = 3
            SA_dim_x = 4
            SA_dim_y = 4
            PART = f"{m},{n},{k}"
            if m > 256 or n > 256 or k > 256: LAT = [16,16]
            else: LAT = [ int(m/SA_dim_x), int(n/SA_dim_y) ]
            LAT = [ str(1) if _ == 0 else str(_) for _ in LAT ]
            LAT = ",".join(LAT)
            SIMD = k if k <= 8 else 4
        # Map reduction loop to space dim
        else:
            ST = 2
            PART = "10,8"
            LAT = "2,8"
            SIMD = 2
            extra_flags += "--local-reduce --reduce-op=\"+\" --simd-touch-space "

    # Params for Conv
    else:
        OC, OH, OW, IC, R, C = loop_bounds
        ST = 4
        print(f"[  INFO  ] input size OC({OC}), OH({OH}), OW({OW}), IC({IC}), R({R}), C({C})")
        PART = "16,15,15,1"
        LAT  = "4,3,3"
        SIMD = "1,1,1,1"
        extra_flags = "--simd-info=./autosa_tests/cnn/simd_info.json "
    return ST, PART, LAT, SIMD, extra_flags

def generate_systolic_array(keys, values, code, backend):
    # Analyze packing and transpose information
    input_attr_info = dict()
    packed_data = list()
    transposed_data = list()

    is_axis_enabled = False
    loop_bounds = list()
    off_chip_data = list()
    written_data = list()

    # Process attribute information for AutoSA module
    for index in range(len(keys)):
        key = keys[index].value
        if key == "axis":
            is_axis_enabled = True
            continue
        elif key == "loop_bound":
            loop_bounds = values[index].value.split(",")
        elif key == "tensor_placement":
            info = values[index].value.split(",")
            for var in info:
                var_name = var.replace("[0]", "").replace("[1]", "")
                var_name = var_name.replace("[read]", "").replace("[write]", "")
                if "[0]" in var:
                    off_chip_data.append(var_name)
                if "[write]" in var:
                    written_data.append(var_name)
        else:
            try:
                is_transpose, pack_factor = values[index].value.split(",")
                input_attr_info[var] = [int(is_transpose), int(pack_factor)]
                if int(pack_factor) > 0:
                    packed_data.append(var)
                if int(is_transpose) == 1:
                    transposed_data.append(var)
            except:
                pass
 
    instance = SystolicArrayRegistry.sa_module_cnt
    autosa_c_source = f"hcl_autosa_tmp_inst{instance}.c"
    pwd = os.getcwd()
    with open(autosa_c_source, "w") as fp:
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

    source_path = os.path.join(pwd, autosa_c_source)
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
    ST, PART, LAT, SIMD, extra_flags = infer_default_params(loop_bounds)
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
        
    cmd += "--hls "
    cmd += "--hcl "
    if is_axis_enabled:
        pass # cmd += "--axi-stream "

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
    cmd += extra_flags

    # addiitonal flags for intel ocl
    if backend == "aocl":
        cmd += "--loop-infinitize --double-buffer-style=0 "

    # Add serialization if the SA module has interface arguements
    cmd += "--host-serialize "
    print(f"[  INFO  ] AutoSA command {cmd}")

    # Save autosa command for debugging purposes
    with open(f"hcl_autosa_cmd_inst{instance}.sh", "w") as fp:
        fp.write(cmd)
    run_process(cmd)
    
    # Extract the autosa generated code
    if backend == "vhls": autosa_header = f"hcl_autosa_tmp_inst{instance}_hcl_decl.h"
    else: autosa_header = "hcl_autosa_tmp_kernel.h"

    ext = "cpp" if backend == "vhls" else "cl"
    source_file = f"{autosa_dir}/autosa.tmp/output/src/hcl_autosa_tmp_inst{instance}_kernel.{ext}"
    with open(source_file, "r") as fp:
        header = fp.read() + "\n"
        header = header.replace(f"#include \"{autosa_header}\"", "")

        if backend == "aocl":
            # Also extract the helper functions for data serialization and deserialization
            with open(f"{autosa_dir}/autosa.tmp/output/src/hcl_autosa_tmp_host.h", "r") as f:
                content = f.read()
                annotation = "/* Helper Function */"
                start_pos = content.find(annotation)
                end_pos = content.rfind(annotation) + len(annotation)
                header += content[start_pos:end_pos] + "\n"
        
        # For xilinx HLS backend
        else:
            count_SA_size(header)

    # External module call inside top function
    with open(f"{autosa_dir}/autosa.tmp/output/src/{autosa_header}", "r") as fp:
        ret_code = fp.readlines()[0].strip() + ";\n"

    # Add prefix to SA functions
    header, ret_code = add_prefix(header, ret_code)

    # Bitcasting the input arguments (to AutoSA selected bit-packing factor)
    # 1. Substitute data type (interface arg) is decided by AutoSA (and possibly do some extra padding).
    # 2. Substitute data serialization size and intrinsic
    ret_code, header = insert_data_pack(ret_code, header, off_chip_data, written_data)

    return [ header, ret_code ] 