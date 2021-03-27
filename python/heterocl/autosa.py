import re

# reinterpret cast orginal pointers to target type
def autosa_infer_types(path, host_code, kernel_code):
    if "/* AutoSA post-processed */" in kernel_code:
        return host_code, kernel_code

    assert kernel_code.find("autosa_func") > 0
    outer = re.compile("void autosa_func\((.*?)\);")
    m = outer.search(kernel_code)
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
        intrinsics = new_ret_code[start_pos:end_pos].replace(annotation, "")
        host_code = host_code.replace(host_start_annotation, intrinsics)
        new_ret_code = new_ret_code[:start_pos] + new_ret_code[end_pos:]

        # Expected serilization in host program
        #  std::vector<float, aligned_allocator<float>> dev_A(4259840);
        #  host_serialize_A(dev_A, A);

    new_ret_code = "/* AutoSA post-processed */\n" + new_ret_code 
    return host_code, new_ret_code