from heterocl.schedule import Stage
from heterocl.types import UInt
from hlib.op.nn import conv2d, max_pool2d
import heterocl as hcl
from heterocl.dsl import def_
import numpy as np
from hlib.op.extern import create_extern_module, register_extern_ip
import os



"""
    Define data types
"""
# SIMD parameters
SIMD_LANE = 8
DATA_W0 = 32
DATA_LDWR_LANE = SIMD_LANE
DEPTH_CONV_LANE = SIMD_LANE
CONV_LANE = SIMD_LANE
RELU_LANE = SIMD_LANE
POOL_LANE = SIMD_LANE
UPSAMPLE_LANE = SIMD_LANE
INTER_LOAD_LANE = SIMD_LANE
INTER_WRITE_LANE = SIMD_LANE
# instruction sizes
CONFIG_PARAMS = 32
INST_PER_LAYER = 5
# data types
data_t0 = hcl.Float(32)  # cin, cout
data_t1 = hcl.Float(32)  # weight
data_t2 = hcl.Float(32)  # bias
data_t3 = hcl.UInt(32)  # inst
DATA_W0 = 32
DATA_W1 = 32
DATA_W2 = 32
DATA_W3 = 32
# bus_t0 = hcl.UInt(512)
# bus_t1 = hcl.UInt(512)
# bus_t2 = hcl.UInt(512)
bus_t0 = hcl.UInt(128)
bus_t1 = hcl.UInt(128)
bus_t2 = hcl.UInt(128)
bus_t3 = hcl.UInt(32)
CinLoadData0Type = hcl.UInt(DATA_W0 * DEPTH_CONV_LANE)
WeightLoadData0Type = hcl.UInt(DATA_W1 * DEPTH_CONV_LANE)
WeigthLoadData1Type = hcl.UInt(DATA_W1 * CONV_LANE)
WeightLoadData2Type = hcl.UInt(DATA_W1 * RELU_LANE)
InterLoadData0Type = hcl.UInt(DATA_W0, INTER_LOAD_LANE)
DepthConvData0Type = hcl.UInt(DATA_W0 * CONV_LANE)
ConvData0Type = hcl.UInt(DATA_W0 * RELU_LANE)
ReluData0Type = hcl.UInt(DATA_W0 * POOL_LANE)
PoolData0Type = hcl.UInt(DATA_W0 * DATA_LDWR_LANE)
InterWriteData0Type = hcl.UInt(DATA_W0 * INTER_WRITE_LANE)
InterWriteData1Type = hcl.UInt(DATA_W0 * INTER_WRITE_LANE)
UpsampleData0Type = hcl.UInt(DATA_W0 * UPSAMPLE_LANE)
ConfigInst = hcl.UInt(192)

def top_module(global_cin, global_prev_cin, global_weight, global_bias, global_cout, config):
    """
        Modules
    """
    # void cin_load(
    # bus_t0 *global_cin, 
    # uint config[CONFIG_PARAMS], 
    # stream<CinLoadData0Type> &fifo_cin, 
    # stream<ConfigInst> &fifo_config_out)
    @register_extern_ip(vendor="xilinx")
    def cin_load(global_cin, config, cin, config_out):
        with hcl.Stage("cin_load") as Module:
            # just put something here
            # or the stage will be optimzied away
            hcl.update(global_cin, lambda *args : global_cin[args]+1, name="update_global_cin")
            hcl.update(config, lambda *args : config[args]+1, name="update_config")
            hcl.update(cin, lambda *args : cin[args]+1, name="update_cin")
            hcl.update(config_out, lambda *args : config_out[args]+1, name="update_config_out")
        Module.ext_ip_name = "cin_load" # top function name
        Module.inputs = [global_cin, config, cin, config_out]
        Module.source = [
            os.path.dirname(os.path.abspath(__file__)) + "/kernel/cin_load.cpp"
        ]
        create_extern_module(Module, ip_type="HLS")
        

    @register_extern_ip(vendor="xilinx")
    def cin_load_prev(global_cin, config_in, cin_prev, config_out):
        with hcl.Stage("cin_load_prev") as Module:
            # just put something here or the stage will be optimized away
            hcl.update(global_cin, lambda *args: global_cin[args] + 1)
            hcl.update(config_in, lambda *args : config_in[args] + 1)
            hcl.update(cin_prev, lambda *args : cin_prev[args] + 1)
            hcl.update(config_out, lambda *args : config_out[args] + 1)
            
        Module.ext_ip_name = "cin_load_prev"
        Module.inputs = [global_cin, config_in, cin_prev, config_out]
        Module.source = [
            os.path.dirname(os.path.abspath(__file__)) + "/kernel/cin_load_prev.cpp"
        ]
        create_extern_module(Module, ip_type="HLS")
        

    @register_extern_ip(vendor="xilinx")
    def weight_load(global_weight, global_bias, config_in, depth_conv_weight, conv_weight, gamma_depth, beta_depth, gamma_conv, beta_conv, config_out):
        with hcl.Stage("weight_load") as Module:
            # just put something here or the stage will be optimized away
            hcl.update(global_weight, lambda *args: global_weight[args] + 1)
            hcl.update(global_bias, lambda *args: global_bias[args] + 1)
            hcl.update(config_in, lambda *x : config_in[x]+1)
            hcl.update(depth_conv_weight, lambda *x : depth_conv_weight[x]+1)
            hcl.update(conv_weight, lambda *x : conv_weight[x]+1)
            hcl.update(gamma_depth, lambda *x : gamma_depth[x]+1)
            hcl.update(beta_depth, lambda *x : beta_depth[x]+1)
            hcl.update(gamma_conv, lambda *x : gamma_conv[x]+1)
            hcl.update(beta_conv, lambda *x : beta_conv[x]+1)
            hcl.update(config_out, lambda *x : config_out[x]+1)
        Module.ext_ip_name = "weight_load"
        Module.inputs = [global_weight, global_bias, config_in, depth_conv_weight, conv_weight, gamma_depth, beta_depth, gamma_conv, beta_conv, config_out]
        Module.source = [
            os.path.dirname(os.path.abspath(__file__)) + "/kernel/weight_load.cpp"
        ]
        create_extern_module(Module, ip_type="HLS")

    @register_extern_ip(vendor="xilinx")
    def cout_write(cout, config_in, global_cout):
        with hcl.Stage("cout_write") as Module:
            hcl.update(cout, lambda *args: cout[args] + 1)
            hcl.update(config_in, lambda *args: config_in[args] + 1)
            hcl.update(global_cout, lambda *args: global_cout[args] + 1)
        Module.ext_ip_name = "cout_write"
        Module.inputs = [cout, config_in, global_cout]
        Module.source = [
            os.path.dirname(os.path.abspath(__file__)) + "/kernel/cout_write.cpp"
        ]
        create_extern_module(Module, ip_type="HLS") 


    @register_extern_ip(vendor="xilinx")
    def conv(cin, weight, config_in, cout, config_out):
        with hcl.Stage("conv") as Module:
            # just put something here
            # or the stage will be optimized away
            hcl.update(cin, lambda *args: cin[args] + 1)
            hcl.update(weight, lambda *args: weight[args] + 1)
            hcl.update(config_in, lambda *x : config_in[x] + 1)
            hcl.update(cout, lambda *x : cout[x] + 1)
            hcl.update(config_out, lambda *x : config_out[x] + 1)
        Module.ext_ip_name = "kernel"  # top function name
        Module.inputs = [cin, weight, cout, config_in, config_out]
        Module.source = [
            os.path.dirname(os.path.abspath(__file__)) + "/kernel/systolic_array.cpp"
        ]
        create_extern_module(Module, ip_type="HLS")

    # change to worst case size
    @def_([(1,63,63,3),(64,3,3,32),(32,),(1,26,26,64),(32,)], dtypes=[DepthConvData0Type, WeightLoadData0Type, ConfigInst, DepthConvData0Type, ConfigInst], name='depth_conv')
    def depth_conv(cin, weight, config_in, cout, config_out):
        """
            this is kernel_size=1 or kernel_size=3 depthwise seperable conv
        """
        # parse layer_config
        en = hcl.compute((1,), lambda _ : config_in[19], dtype=hcl.UInt(32))
        layer_en = hcl.unpack(en, dtype=hcl.UInt(1))
        DEPTH_CONV_EN = hcl.compute((1,), lambda _ : layer_en[1])
        
        FILTER_S = hcl.compute((1,), lambda _ : config_in[16], dtype=hcl.UInt(32))
        FILTER_S1_S2 = hcl.unpack(FILTER_S, dtype=hcl.UInt(16))
        FILTER_S1 = hcl.compute((1,), lambda _ : FILTER_S1_S2[0], "FILTER_S1")

        LAYER_IN_NUM = hcl.compute((1,), lambda _ : config_in[6], dtype=hcl.UInt(32))

        # write config_out
        config_out = hcl.compute(config_in.shape, lambda *args : config_in[args])

        with hcl.if_(DEPTH_CONV_EN):
            with hcl.if_(FILTER_S1): 
                # kernel size should be 1x1
                # TODO: why padding=[0,0] failed? 
                cout = conv2d(cin, weight, padding=[1,1], groups=1, data_layout='NHWC')
            with hcl.else_():
                # kernel size should be 3x3
                cout = conv2d(cin, weight, padding=[1,1], groups=hcl.cast(hcl.Int(32), LAYER_IN_NUM), data_layout='NHWC')
        with hcl.else_():
            cout = hcl.compute(cin.shape, lambda *args : cin[args]) 



    # why didn't they use pool???
    @def_([(1,26,26,3), (32,), (1,13,13,3), (32,)], name="pool")
    def pool(cin, config_in, cout, config_out):
        # parse layer_config
        en = hcl.compute((1,), lambda _ : config_in[19], dtype=hcl.UInt(32))
        layer_en = hcl.unpack(en, dtype=hcl.UInt(1))
        pool_en = hcl.compute((1,), lambda _ : layer_en[5], dtype=hcl.UInt(1))

        # write config_out
        config_out = hcl.compute(config_in.shape, lambda *args : config_in[args])

        # perform pooling or bypass
        with hcl.if_(pool_en):
            cout = max_pool2d(
                cin, pool_size=[2, 2], strides=[2, 2], padding=[0, 0], layout="NHWC", name="pool"
            )
        with hcl.else_():
            cout = hcl.compute(cin.shape, lambda *args : cin[args])

    @def_([(1,26,26,3), (32,), (1,26,26,3), (32,), (1,), (1,)], dtypes=[DepthConvData0Type, ConfigInst, ReluData0Type, ConfigInst, CinLoadData0Type, CinLoadData0Type], name="relu_bn")
    def relu(cin, config_in, cout, config_out, gamma_conv, beta_conv):
        # parse layer_config
        en = hcl.compute((1,), lambda _ : config_in[19], dtype=hcl.UInt(32))
        layer_en = hcl.unpack(en, dtype=hcl.UInt(1))
        CONV_EN = hcl.compute((1,), lambda _ : layer_en[2], dtype=hcl.UInt(1)) 
        RELU_EN = hcl.compute((1,), lambda _ : layer_en[3], dtype=hcl.UInt(1))
        RELU6_EN = hcl.compute((1,), lambda _ : layer_en[4], dtype=hcl.UInt(1))
        BIAS_EN = hcl.compute((1,), lambda _ : layer_en[7], dtype=hcl.UInt(1))
        BATCH_NORM_EN = hcl.compute((1,), lambda _ : layer_en[10], dtype=hcl.UInt(1))
        BATCH_NORM_EN_DEPTH = hcl.compute((1,), lambda _ : layer_en[12], dtype=hcl.UInt(1))
        cond = hcl.or_(RELU_EN, RELU6_EN, BIAS_EN, BATCH_NORM_EN)

        # write config_out
        config_out = hcl.compute(config_in.shape, lambda *args : config_in[args])

        with hcl.if_(cond):
            bias_en = hcl.and_(BIAS_EN, CONV_EN)
            with hcl.if_(hcl.or_(bias_en, BATCH_NORM_EN)):
                cout = hcl.compute(cin.shape, lambda *args : cin[args] * gamma_conv + beta_conv)
            with hcl.if_(hcl.and_(RELU6_EN, BATCH_NORM_EN_DEPTH == 0)):
                cout = hcl.compute(cout.shape, lambda *args : hcl.select(cin[args] > 6, hcl.cast(cin.dtype, 6), cin[args]))
            with hcl.elif_(RELU_EN):
                cout = hcl.compute(cin.shape, lambda *args : hcl.select(cin[args] < 0, hcl.cast(cin.dtype, 0), cin[args]))
        with hcl.else_():
            cout = hcl.compute(cin.shape, lambda *args : cin[args])



    # not available in hlib.nn
    # bilinear is more complicated, we implement nearest neighbor first
    # actualy we can use conv2d_tranpose to implement bilinear
    @def_([(1,26,26,3),(32,),(1,52,52,3),(32,)], dtypes=[ReluData0Type, ConfigInst, UpsampleData0Type, ConfigInst], name="nearest_neighbor_upsample")
    def upsample(cin, config_in, cout, config_out):
        # parse layer_config
        en = hcl.compute((1,), lambda _ : config_in[19], dtype=hcl.UInt(32))
        layer_en = hcl.unpack(en, dtype=hcl.UInt(1))
        UP_SAMPLE_EN = hcl.compute((1,), lambda _ : layer_en[6])

        # write config_out
        config_out = hcl.compute(config_in.shape, lambda *args : config_in[args])

        with hcl.if_(UP_SAMPLE_EN):
            n, h, w, c = cin.shape
            # TODO: is this ok?
            cout = hcl.compute((n, h*2, w*2, c), lambda n_i,h_i,w_i,c_i : cin[n_i][h_i/2][w_i/2][c_i])
        with hcl.else_():
            cout = hcl.compute(cin.shape, lambda *args : cin[args]) 


    # not availabel in hlib.nn
    @def_([(1,26,26,3), (1,26,26,3), (32,), (1,26,26,3), (32,)], dtypes=[CinLoadData0Type, ReluData0Type, ConfigInst, ReluData0Type, ConfigInst], name="add")
    def add(cin1, cin2, config_in, cout, config_out):
        # parse layer_config
        en = hcl.compute((1,), lambda _ : config_in[19], dtype=hcl.UInt(32))
        layer_en = hcl.unpack(en, dtype=hcl.UInt(1))
        LOAD_PREV_EN = hcl.compute((1,), lambda _ : layer_en[11]) 
        
        # write config_out
        config_out = hcl.compute(config_in.shape, lambda *args : config_in[args])

        with hcl.if_(LOAD_PREV_EN):
            cout = hcl.compute(cin1.shape, lambda *args : cin1[args] + cin2[args])
        with hcl.else_():
            cout = hcl.compute(cin2.shape, lambda *args : cin2[args])
                

    """
        Top module
    """


    def engine(
        global_cin, global_prev_cin, global_weight, global_bias, global_cout, layer_config
    ):
        """
        pass one layer
        """
        # TODO: How to determine the sizes?
        # this question is actually a bit complicated? Because FlexCNN processes one block each run, so
        # the input shapes are not always the same. But we don't have dynamic-shape tensors in HeteroCL.

        # One thing I would like to try:
        # Generate pointers/stream objects for function interfaces.
        # Then we can control read/write access
        # fifo_cin_load_0 = hcl.compute(())

        # We don't know the sizes yet
        cin_load_0          = hcl.compute((1000,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_cin_load_0")
        weight_load_0       = hcl.compute((1000,), lambda *_ : 0, dtype=WeightLoadData0Type, name="fifo_weight_load_0")
        weight_load_1       = hcl.compute((1000,), lambda *_ : 0, dtype=WeightLoadData0Type, name="fifo_weight_load_1")
        depth_conv_0        = hcl.compute((1000,), lambda *_ : 0, dtype=DepthConvData0Type, name="fifo_depth_conv_0")
        relu6_0             = hcl.compute((1000,), lambda *_ : 0, dtype=ReluData0Type, name="fifo_relu6_0")
        conv_0              = hcl.compute((1000,), lambda *_ : 0, dtype=DepthConvData0Type, name="fifo_conv_0")
        add_0               = hcl.compute((1000,), lambda *_ : 0, dtype=ReluData0Type, name="fifo_add_0")
        relu_0              = hcl.compute((1000,), lambda *_ : 0, dtype=ReluData0Type, name="fifo_relu_0")
        upsample_0          = hcl.compute((1000,), lambda *_ : 0, dtype=UpsampleData0Type, name="fifo_upsample_0")
        upsample_1          = hcl.compute((1000,), lambda *_ : 0, dtype=UpsampleData0Type, name="fifo_upsample_1")
        merge_0             = hcl.compute((1000,), lambda *_ : 0, dtype=UpsampleData0Type, name="fifo_merge_0")
        cin_prev_0          = hcl.compute((1000,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_cin_prev_0")
        beta_depth          = hcl.compute((1000,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_beta_depth")
        gamma_depth         = hcl.compute((1000,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_gamma_depth")
        beta_conv           = hcl.compute((1000,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_beta_conv")
        gamma_conv          = hcl.compute((1000,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_gamma_conv")
        config_prev_load    = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_prev_load")
        config_weight_load  = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_weight_load")
        config_depth_conv   = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_depth_conv")
        config_relu6        = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_relu6")
        config_conv         = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_conv")
        config_add          = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_add")
        config_relu         = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_relu")
        config_upsample     = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_upsample")
        config_merge        = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name='config_merge')
        config_data_write   = hcl.compute((1000,), lambda *_ : 0, dtype=ConfigInst, name="config_data_write")

        # connect modules 
        cin_load(global_cin, layer_config, cin_load_0, config_prev_load)
        cin_load_prev(global_prev_cin, config_prev_load, cin_prev_0, config_weight_load)
        weight_load(global_weight, global_bias, config_weight_load, weight_load_0, weight_load_1, beta_depth, gamma_depth, beta_conv, gamma_conv, config_depth_conv)
        depth_conv(cin_load_0, weight_load_0, config_depth_conv, depth_conv_0, config_relu6)
        relu(depth_conv_0, config_relu6, relu6_0, config_conv, beta_depth, gamma_depth)
        conv(relu6_0, weight_load_1, config_conv, conv_0, config_relu)
        relu(conv_0, config_relu, relu6_0, config_add, beta_conv, gamma_conv)
        add(cin_prev_0, relu_0, config_add, add_0, config_upsample)
        upsample(add_0, config_upsample, upsample_0, merge_0, config_merge) # flexcnn's upsample+merge_upsample
        cout_write(merge_0, config_data_write, global_cout)


    """
    call engine until all layers have passed
    """
    with hcl.Stage("top_kernel"):
        # 87 layers
        with hcl.for_(0, 86) as layer_id:
            layer_config = hcl.compute(
                (32,), lambda x: config[x + layer_id * 32], "layer_config"
            )
            engine(
                global_cin,
                global_prev_cin,
                global_weight,
                global_bias,
                global_cout,
                layer_config
            )



def test_flexcnn():
    hcl.init(hcl.Float())

    LAYER_NUM = 87
    CONFIG_PARAMS = 32

    # the sizes are from SDx_project/src/params.h
    # TODO: could these sizes be too big?
    global_cin = hcl.placeholder((12625160,), name="global_cin", dtype=bus_t0)
    global_prev_cin = hcl.placeholder(
        (12625160,), name="global_prev_cin", dtype=bus_t0
    )
    global_weight = hcl.placeholder(
        (560032,), name="global_weight", dtype=bus_t1
    )
    global_bias = hcl.placeholder((16544,), name="global_bias", dtype=bus_t2)
    global_cout = hcl.placeholder(
        (826274,), name="global_cout", dtype=bus_t0
    )  # unsure of this size
    config = hcl.placeholder(
        (5 + LAYER_NUM * CONFIG_PARAMS,), name="config", dtype=bus_t3
    )

    arg_list = [
        global_cin,
        global_prev_cin,
        global_weight,
        global_bias,
        global_cout,
        config,
    ]

    s = hcl.create_schedule(arg_list, top_module)
    p = hcl.platform.aws_f1
    p.config(compile='vitis', mode='debug')
    # p = "vhls"

    code = str(hcl.build(s, p, name="main"))
    print(hcl.lower(s))
    with open("flexcnn.cpp", "w") as f:
        f.write(code)


if __name__ == "__main__":
    test_flexcnn()
