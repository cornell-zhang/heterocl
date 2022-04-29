import heterocl as hcl
import hlib.bnn as bnn
import numpy as np
import sys

target = hcl.Platform.xilinx_zc706
target.config(compiler="vivado_hls", mode="csim|csyn", project="bnn-nhwc.prj")
test_size = 100
batch_size = 100
qtype_bit = hcl.UInt(1)  # weights
qtype_int = hcl.Int(6)  # not unsigned!
qtype_float = hcl.Fixed(20, 10)
qtype_packed = hcl.UInt(32)

dtype_in = qtype_bit
dtype_out = qtype_float

# prepare numpy arrays for testing
data = np.load("data/bnn-5775.data.npz")
images = data["images"][:test_size]
labels = data["labels"][:test_size]
num_images = images.shape[0]
params = np.load("data/bnn-5775.params.npz")

# prepare packed arrays
packed_params = {}
for name in params:
    if "w_fc" in name:
        packed_params[name] = np.packbits(params[name].copy().astype(np.bool),
            axis=1,bitorder="little").view(np.uint32)
    elif "w_conv1" in name:
        arr = params[name].copy().transpose(0,2,3,1).astype(np.bool)
        packed_params[name] = arr
    elif "w_conv2" in name:
        arr = params[name].copy().transpose(0,2,3,1)
        arr = np.packbits(arr.astype(np.bool),
                axis=3,bitorder="little").view(np.uint16)
        packed_params[name] = arr
    elif "bn_t" in name:
        packed_params[name] = params[name].copy().transpose(1,2,0)
    else:
        packed_params[name] = params[name].copy()
    # print(name, packed_params[name].shape, packed_params[name].dtype)

# def build_packed_bnn(input_image):
#     w_conv1 = hcl.const_tensor(packed_params["w_conv1"],"w_conv1",qtype_bit)
#     bn_t1 = hcl.const_tensor(packed_params["bn_t1"],"bn_t1",qtype_float)
#     w_conv2 = hcl.const_tensor(packed_params["w_conv2"],"w_conv2",hcl.UInt(16))
#     bn_t2 = hcl.const_tensor(packed_params["bn_t2"],"bn_t2",qtype_float)
#     w_fc1 = hcl.const_tensor(packed_params["w_fc1"],"w_fc1",qtype_packed)
#     b_fc1 = hcl.const_tensor(packed_params["b_fc1"],"b_fc1",qtype_float)
#     w_fc2 = hcl.const_tensor(packed_params["w_fc2"],"w_fc2",qtype_packed)
#     b_fc2 = hcl.const_tensor(packed_params["b_fc2"],"b_fc2",qtype_float)
def build_packed_bnn(input_image, w_conv1, bn_t1,
              w_conv2, bn_t2,
              w_fc1, b_fc1,
              w_fc2, b_fc2):  # 1*16*16
    conv1 = bnn.packed_conv2d_nhwc(input_image, w_conv1, padding=[1,1], name="conv1", out_dtype=qtype_int)
    bn1 = bnn.packed_batch_norm_threshold_nhwc(conv1, bn_t1, name="bn1")
    maxpool1 = bnn.packed_max_pool2d_nhwc(bn1, [2,2], [2,2], name="maxpool1")

    conv2 = bnn.packed_conv2d_nhwc(maxpool1, w_conv2, padding=[1,1], name="conv2", out_dtype=qtype_int)
    bn2 = bnn.packed_batch_norm_threshold_nhwc(conv2, bn_t2, name="bn2")
    maxpool2 = bnn.packed_max_pool2d_nhwc(bn2, [2,2], [2,2], name="maxpool2") # 32*4*4=512

    pack = bnn.packed_flatten_nhwc(maxpool2,name="packed_flatten")
    fc1 = bnn.packed_dense(pack, w_fc1, b_fc1, True, name="fc1", dtype=hcl.UInt(32)) # 512/32->256/32
    fc2 = bnn.packed_dense(fc1, w_fc2, b_fc2, False, name="fc2", dtype=dtype_out) # 256/32->10
    return fc2

def build_bitpacked_bnn_inf(batch_size=batch_size,target=target):
    # print("build_bitpacked_bnn_inf")
    # input_image = hcl.placeholder((batch_size,16,16,1),"input_image",dtype_in)
    # s = hcl.create_schedule([input_image], build_packed_bnn)
    # return hcl.build(s, target=target)

    hcl_ph = []
    input_image = hcl.placeholder(
        (batch_size, 16, 16, 1), "input_image", dtype_in)
    hcl_ph.append(hcl.placeholder((16, 3, 3, 1), "w_conv1", qtype_bit))
    hcl_ph.append(hcl.placeholder((16, 16, 16), "bn_t1", qtype_float))
    hcl_ph.append(hcl.placeholder((32, 3, 3, 1), "w_conv2", hcl.UInt(16)))
    hcl_ph.append(hcl.placeholder((8, 8, 32), "bn_t2", qtype_float))
    hcl_ph.append(hcl.placeholder((256, 16), "w_fc1", hcl.UInt(32)))
    hcl_ph.append(hcl.placeholder((256,), "b_fc1", qtype_float))
    hcl_ph.append(hcl.placeholder((10, 8), "w_fc2", hcl.UInt(32)))
    hcl_ph.append(hcl.placeholder((10,), "b_fc2", qtype_float))
    s = hcl.create_schedule([input_image] + hcl_ph, build_packed_bnn)
    print(s.device_module)

    return hcl.build(s, target=target)

if __name__ == '__main__':

    f = build_bitpacked_bnn_inf(batch_size, target)
    # with open("bnn-nhwc.prj/kernel.cpp", "w") as outfile:
    #     outfile.write(f)
    # hcl.execute_fpga_backend(target)
    f()
