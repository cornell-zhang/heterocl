import re

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
        host_code = host_code.replace(f"{size}, {tensor}", f"{new_size}, {tensor}", 1)
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
    outer = re.compile("void autosa_func\((.*?)\);")
    m = outer.search(kernel_code)
    assert m is not None
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