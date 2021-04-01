import heterocl as hcl
import numpy as np

N_HASH_FUNC = 5
HASH_RANGE = 10
PRIME_NUM = 39916801
RAND_SEED = 1000
PACKET_SIZE = 10

UPDATE = 1
QUERY = 2
EXIT = 3

def func_wrapper(dtype):
    hcl.init(dtype)
    
    a=hcl.placeholder((N_HASH_FUNC,), dtype=dtype, name="a")
    b=hcl.placeholder((N_HASH_FUNC,), dtype=dtype, name="b")
    key=hcl.placeholder((PACKET_SIZE,), dtype=dtype, name="key")
    result=hcl.placeholder((PACKET_SIZE,), dtype=dtype, name="result")
    mode = hcl.placeholder((PACKET_SIZE, ), name="mode")

    def hash_index(mode, a, b, key, result):
        sketch = hcl.compute((N_HASH_FUNC, HASH_RANGE), lambda x,y: 0, name="sketch")
        with hcl.for_(0, PACKET_SIZE) as p:
            with hcl.if_(mode[p] == UPDATE):
                key_local = hcl.compute((1,), lambda x: key[p], name = "key_local")
                index = hcl.compute(a.shape, lambda x: (a[x]*key_local[0] + b[x])%HASH_RANGE, name="index")
                with hcl.for_(0, N_HASH_FUNC) as i:
                    sketch[i][index[i]] += 1
                min_value = hcl.compute((1,), lambda x: sketch[0][index[0]],name="min_val")
                with hcl.for_(1, N_HASH_FUNC) as n:
                    with hcl.if_(min_value[0] > sketch[n][index[n]]):
                        min_value[0] = sketch[n][index[n]]
                result[p] = min_value[0]
            with hcl.if_(mode[p] == QUERY):
                key_local = hcl.compute((1,), lambda x: key[p],name = "key_local")
                index = hcl.compute(a.shape, lambda x: (a[x]*key_local[0] + b[x])%HASH_RANGE,name="index")
                min_value = hcl.compute((1,), lambda x: sketch[0][index[0]],name="min_val")
                with hcl.for_(1, N_HASH_FUNC) as n:
                    with hcl.if_(min_value[0] > sketch[n][index[n]]):
                        min_value[0] = sketch[n][index[n]]
                result[p] = min_value[0]
            with hcl.if_(mode[p] == EXIT):
                result[p] = -1
                hcl.break_()
    
    s = hcl.create_schedule([mode, a, b, key, result], hash_index)
    # print(hcl.lower(s))
    # target = "llvm"
    # target = hcl.platform.aws_f1
    # target.config(compile='vitis', mode='sw_sim')
    # config = {"host": hcl.dev.asic("mentor"), "xcel": [hcl.dev.asic("mentor")]}
    # target = hcl.platform.custom(config)
    target = hcl.Platform.asic_hls
    target.config(compile="catapultc", mode="sw_sim")
    s.to([mode, a, b], target.xcel, mode=hcl.IO.DMA)
    s.to(key, target.xcel, mode=hcl.IO.Stream)
    s.to(result, target.host, mode=hcl.IO.Stream)
    f = hcl.build(s, target, name='countmin')
    # print(f)

    np_a = np.random.randint(10, size=(N_HASH_FUNC,))
    np_b = np.random.randint(10, size=(N_HASH_FUNC,))
    np_key = np.random.randint(10, size=(PACKET_SIZE,))
    np_result = np.zeros(PACKET_SIZE)
    np_mode = np.array([1,1,1,1,1,2,2,2,2,3])
    args = (np_mode, np_a, np_b, np_key, np_result)
    f.inspect(args)
    f.execute(args)


    # hcl_a = hcl.asarray(np.random.randint(10, size=(N_HASH_FUNC,)), dtype=dtype)
    # # hcl_a = hcl.asarray(np.array([7,3,3,1,7]), dtype=dtype)
    # hcl_b = hcl.asarray(np.random.randint(10, size=(N_HASH_FUNC,)), dtype=dtype)
    # # hcl_b = hcl.asarray(np.array([9,9,4,1,2]), dtype=dtype)
    # hcl_key = hcl.asarray(np.random.randint(10, size=(PACKET_SIZE,)), dtype=dtype)
    # hcl_result = hcl.asarray(np.zeros(PACKET_SIZE), dtype=dtype)
    # hcl_mode = hcl.asarray(np.array([1,1,1,1,1,2,2,2,2,3]), dtype=dtype)
    # np_a = hcl_a.asnumpy()
    # np_b = hcl_b.asnumpy()
    # np_key = hcl_key.asnumpy()
    # np_result = hcl_result.asnumpy()
    # np_mode = hcl_mode.asnumpy()
    # print("a:", np_a)
    # print("b:", np_b)
    # print("key:", np_key)
    # print("result:", np_result)
    # print("mode:", np_mode)

    # f(hcl_mode, hcl_a, hcl_b, hcl_key, hcl_result)
    # np_a = hcl_a.asnumpy()
    # np_b = hcl_b.asnumpy()
    # np_key = hcl_key.asnumpy()
    # np_result = hcl_result.asnumpy()
    # np_mode = hcl_mode.asnumpy()
    # print("a:", np_a)
    # print("b:", np_b)
    # print("key:", np_key)
    # print("result:", np_result)
    # print("mode:", np_mode)

if __name__ == "__main__":
    func_wrapper(hcl.Int(32))
