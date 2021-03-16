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
    for pair in pairs:
        _ = pair.split(" *")
        dtype, tensor = _
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

    new_ret_code = "/* AutoSA post-processed */\n" + new_ret_code 
    return host_code, new_ret_code