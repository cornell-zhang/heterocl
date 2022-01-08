import heterocl as hcl
st = hcl.Struct({"key": hcl.Int(32), "val": hcl.Int(32)})
size = 1024
class_number  = 6
compute_units = 4

hcl.init(hcl.UInt(32))
inputs = hcl.placeholder((size,), dtype=st, name="input")

def kernel(inputs):
    
    def split(inputs, number):
        cus = []
        size = inputs.shape[0]
        for i in range(number):
            base = i * (size/number)
            name = "batch_" + str(i)
            ret = hcl.compute((int(size/number),), 
                    lambda x: inputs[base+x], dtype=st, name=name)
            cus.append(ret)
        return cus
    
    # ret is the input slice { (key, value)...}
    # res is the intermediate result
    def count(res, ret, x):
        res[ret[x].key] += ret[x].val
    
    def reducer(ress, output, x):
        for res in ress:
            output[x] += res[x]

    rets = split(inputs, compute_units)

    ress = []
    for ret in rets:
        name = "map_batch_" + str(rets.index(ret))
        res = hcl.compute((class_number,), lambda *args: 0, name=name)
        # mapping (accumulate quality scores in each batch)
        hcl.mutate((int(size/compute_units),), 
                lambda x: count(res, ret, x), name="mutate_" + name)
        ress.append(res)
    
    # shuffle and reduce the ress into output
    output = hcl.compute((class_number, ), lambda x: 0, name="output")
    hcl.mutate((class_number,), lambda x: reducer(ress, output, x), "reducer")
    return output

target = hcl.platform.aws_f1
s = hcl.create_schedule([inputs], kernel)

# new_inputs = s.to(inputs, target.xcel)
# s.to(kernel.reducer.output, target.host)

# consumers = [getattr(kernel, "batch_" + str(_)) for _ in range(compute_units)]
# s.multicast(new_inputs, consumers)
print(hcl.lower(s))
