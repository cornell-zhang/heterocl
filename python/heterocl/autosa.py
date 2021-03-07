import re

# reinterpret cast orginal pointers to target type
def autosa_infer_types(header, ret_code):
    assert header.find("autosa_func") > 0
    outer = re.compile("autosa_func\((.*?)\)")
    m = outer.search(header)
    inner_str = m.group(1)

    # find inner pairs
    target_dtype = dict()
    pairs = inner_str.split(", ")
    for pair in pairs:
        _ = pair.split(" *")
        dtype, tensor = _
        target_dtype[tensor] = dtype
    
    # replace the original pointers
    new_ret_code = ret_code
    for k, v in target_dtype.items():
        new_ret_code = new_ret_code.\
            replace("buffer_{}[0]".format(k), \
            "reinterpret_cast<{}*>({})".format(v, k))
    return new_ret_code