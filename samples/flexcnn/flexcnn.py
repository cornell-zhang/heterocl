from heterocl.schedule import Stage
from heterocl.types import UInt
from hlib.op.nn import conv2d, max_pool2d
import heterocl as hcl
from heterocl.dsl import def_
import numpy as np
from hlib.op.extern import create_extern_module, register_extern_ip
import os
from utils import *
from config import *

"""
    For function call stack visualization
"""
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
graphviz = GraphvizOutput()
graphviz.output_file = 'flexcnn_callgraph.pdf'
graphviz.output_type = 'pdf'
# config = Config(max_depth=5)


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
bus_t0 = hcl.UInt(512)
bus_t1 = hcl.UInt(512)
bus_t2 = hcl.UInt(512)
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
        Extern Modules
    """
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

    """
        Utility functions
    """
    def decode_instructions(config_in):
        # 192-bit
        inst0 = hcl.Struct({
            'in_num_hw'  : hcl.UInt(32),
            'out_num_hw' : hcl.UInt(32),
            'in_h_hw'    : hcl.UInt(32),
            'in_w_hw'    : hcl.UInt(32),
            'out_h_hw'   : hcl.UInt(32),
            'out_w_hw'   : hcl.UInt(32)
        })
        # 192-bit
        inst1 = hcl.Struct({
            'in_num'    : hcl.UInt(32),
            'out_num'   : hcl.UInt(32),
            'in_h'      : hcl.UInt(32),
            'in_w'      : hcl.UInt(32),
            'out_h'     : hcl.UInt(32),
            'out_w'     : hcl.UInt(32)
        })
        # 192-bit
        inst2 = hcl.Struct({
            'cin_offset'    : hcl.UInt(32),
            'weight_offset' : hcl.UInt(32),
            'bias_offset'   : hcl.UInt(32),
            'cout_offset'   : hcl.UInt(32),
            'filter_s1'     : hcl.UInt(16),
            'filter_s2'     : hcl.UInt(16),
            'stride'        : hcl.UInt(32)
        })
        # 192-bit
        inst3 = hcl.Struct({
            'layer_en'          : hcl.UInt(32),
            'prev_cin_offset'   : hcl.UInt(32),
            'in_num_t'          : hcl.UInt(16),
            'out_num_t'         : hcl.UInt(16),
            'in_h_t'            : hcl.UInt(32),
            'in_w_t'            : hcl.UInt(32),
            'nxt_layer_batch'   : hcl.UInt(32)
        })
        # 192-bit
        inst4 = hcl.Struct({
            'task_num1'       : hcl.UInt(32),
            'task_num2'       : hcl.UInt(32),
            'local_accum_num' : hcl.UInt(32),
            'local_reg_num'   : hcl.UInt(32),
            'row_il_factor'   : hcl.UInt(32),
            'col_il_factor'   : hcl.UInt(32)
        })
        inst0.in_num_hw       = hcl.Scalar(config_in[0][32*0:32*1])
        inst0.out_num_hw      = hcl.Scalar(config_in[0][32*1:32*2])
        inst0.in_h_hw         = hcl.Scalar(config_in[0][32*2:32*3])
        inst0.in_w_hw         = hcl.Scalar(config_in[0][32*3:32*4])
        inst0.out_h_hw        = hcl.Scalar(config_in[0][32*4:32*5])
        inst0.out_w_hw        = hcl.Scalar(config_in[0][32*5:32*6])
        inst1.in_num          = hcl.Scalar(config_in[1][32*0:32*1])
        inst1.out_num         = hcl.Scalar(config_in[1][32*1:32*2])
        inst1.in_h            = hcl.Scalar(config_in[1][32*2:32*3])
        inst1.in_w            = hcl.Scalar(config_in[1][32*3:32*4])
        inst1.out_h           = hcl.Scalar(config_in[1][32*4:32*5])
        inst1.out_w           = hcl.Scalar(config_in[1][32*5:32*6])
        inst2.cin_offset      = hcl.Scalar(config_in[2][32*0:32*1])
        inst2.weight_offset   = hcl.Scalar(config_in[2][32*1:32*2])
        inst2.bias_offset     = hcl.Scalar(config_in[2][32*2:32*3])
        inst2.cout_offset     = hcl.Scalar(config_in[2][32*3:32*4])
        inst2.filter_s1       = hcl.Scalar(config_in[2][32*4:32*4+16])
        inst2.filter_s2       = hcl.Scalar(config_in[2][32*4+16:32*5])
        inst2.stride          = hcl.Scalar(config_in[2][32*5:32*6])
        inst3.layer_en        = hcl.Scalar(config_in[3][32*0:32*1])
        inst3.prev_cin_offset = hcl.Scalar(config_in[3][32*1:32*2])
        inst3.in_num_t        = hcl.Scalar(config_in[3][32*2:32*2+16])
        inst3.out_num_t       = hcl.Scalar(config_in[3][32*2+16:32*3])
        inst3.in_h_t          = hcl.Scalar(config_in[3][32*3:32*4])
        inst3.in_w_t          = hcl.Scalar(config_in[3][32*4:32*5])
        inst3.nxt_layer_batch = hcl.Scalar(config_in[3][32*5:32*6])
        inst4.task_num1       = hcl.Scalar(config_in[4][32*0:32*1])
        inst4.task_num2       = hcl.Scalar(config_in[4][32*1:32*2])
        inst4.local_accum_num = hcl.Scalar(config_in[4][32*2:32*3])
        inst4.local_reg_num   = hcl.Scalar(config_in[4][32*3:32*4])
        inst4.row_il_factor   = hcl.Scalar(config_in[4][32*4:32*5])
        inst4.col_il_factor   = hcl.Scalar(config_in[4][32*5:32*6])

        return inst0, inst1, inst2, inst3, inst4


    """
        HeteroCL-native modules
    """
    # NHWC layout
    @def_([(DEPTH_CIN_BUFFER_SIZE,), (DEPTH_WEIGHT_BUFFER_SIZE,),(CONFIG_BUFFER_SIZE,),(DEPTH_COUT_BUFFER_SIZE,),(CONFIG_BUFFER_SIZE,)], dtypes=[DepthConvData0Type, WeightLoadData0Type, ConfigInst, DepthConvData0Type, ConfigInst], name='depth_conv')
    def depth_conv(cin, weight, config_in, cout, config_out):
        # write config_out
        config_out = hcl.compute((1,), lambda *args : config_in[args])
        # decode instruction
        inst0, inst1, inst2, inst3, inst4 = decode_instructions(config_in)
        # set up enable signals
        DEPTH_CONV_EN = hcl.compute((1,), lambda _ : inst3.layer_en[1], dtype=hcl.UInt(1))
        LAYER_IN_NUM_HW     = inst0.in_num_hw.asnode()
        LAYER_OUT_NUM_HW    = inst0.out_num_hw.asnode()
        LAYER_IN_H_HW       = inst0.in_h_hw.asnode()
        LAYER_IN_W_HW       = inst0.in_w_hw.asnode()
        LAYER_IN_NUM_T      = inst3.in_num_t.asnode()
        LAYER_OUT_NUM_T     = inst3.out_num_t.asnode()
        LAYER_IN_H_T        = inst3.in_h_t.asnode()
        LAYER_IN_W_T        = inst3.in_w_t.asnode()
        STRIDE              = inst2.stride.asnode()
        FILTER_S1           = inst2.filter_s1.asnode()
        
        kernel_size = FILTER_S1
        
        with hcl.if_(DEPTH_CONV_EN):
            with hcl.for_(0, LAYER_OUT_NUM_HW / LAYER_OUT_NUM_T) as out_num_iter:
                with hcl.for_(0, LAYER_IN_W_HW / LAYER_IN_W_T)      as in_w_iter:
                    with hcl.for_(0, LAYER_IN_H_HW / LAYER_IN_H_T)     as in_h_iter:
                        with hcl.for_(0, LAYER_IN_NUM_HW / LAYER_IN_NUM_T) as in_num_iter:
                            """ for each tile """
                            with hcl.for_(0, LAYER_IN_NUM_T / DEPTH_CONV_LANE) as o:
                                with hcl.for_(0, LAYER_IN_H_T / STRIDE) as h:
                                    with hcl.for_(0, LAYER_IN_W_T / STRIDE) as w:
                                        # calculate index
                                        in_indices, out_index = depth_calc_inout_idx(
                                            o_idx=o,
                                            h_idx=h,
                                            w_idx=w,
                                            h_idx_t=in_h_iter,
                                            w_idx_t=in_w_iter,
                                            c_idx_t=in_num_iter,
                                            h_t=LAYER_IN_H_T,
                                            w_t=LAYER_IN_W_T,
                                            c_t=LAYER_IN_NUM_T,
                                            h_hw=LAYER_IN_H_HW,
                                            w_hw=LAYER_IN_W_HW,
                                            c_hw=LAYER_IN_NUM_HW,
                                            cout_hw=LAYER_OUT_NUM_HW,
                                            kernel_size=kernel_size,
                                            stride=STRIDE
                                        )
                                        w_indices = depth_calc_weights_idx(
                                            cout_idx=out_num_iter,
                                            kernel_size=kernel_size,
                                            channel_in=LAYER_IN_NUM_HW/LAYER_IN_NUM_T,
                                            channel_index=o
                                        )
                                        cout_float = hcl.compute((DEPTH_CONV_LANE,), lambda *args : 0, name="depth_float")
                                        def conv_SIMD(idx):
                                            # fetch cin and weights
                                            cin_index = in_indices[idx]
                                            w_index = w_indices[idx]
                                            # fetch data and handle padding 
                                            # TODO: handle padding
                                            cin_SIMD = hcl.compute((1,), lambda *_ : cin[cin_index], name="cin_SIMD") 
                                            w_SIMD   = hcl.compute((1,), lambda *_ : weight[w_index], name="w_SIMD")
                                            # unpack
                                            cin_uint = hcl.unpack(cin_SIMD, dtype=hcl.UInt(32))
                                            w_uint   = hcl.unpack(w_SIMD,   dtype=hcl.UInt(32))
                                            # bitcast to float
                                            cin_float = hcl.bitcast(cin_uint, dst_dtype=hcl.Float(32))
                                            w_float   = hcl.bitcast(w_uint,   dst_dtype=hcl.Float(32))
                                            # calculate
                                            hcl.update(cout_float, lambda *args : cout_float[args] + cin_float[args] * w_float[args])
                                        hcl.mutate((kernel_size*kernel_size,), lambda x : conv_SIMD(x))
                                        # bitcast back to uint
                                        cout_uint = hcl.bitcast(cout_float, hcl.UInt(32))
                                        # pack
                                        cout_SIMD = hcl.pack(cout_uint, dtype=DepthConvData0Type)
                                        cout[out_index] = cout_SIMD
        with hcl.else_():
            hcl.update(cout, lambda *args : cin[args])




    # FlexCNN's open-source design did not use Pooling
    # because OpenPose does not have pooling layer
    @def_([(POOL_CIN_BUFFER_SIZE,), (CONFIG_BUFFER_SIZE,), (POOL_COUT_BUFFER_SIZE,), (CONFIG_BUFFER_SIZE,)], name="pool")
    def pool(cin, config_in, cout, config_out):
        pass

    @def_([(RELU6_CIN_BUFFER_SIZE,), (CONFIG_BUFFER_SIZE,), (RELU6_COUT_BUFFER_SIZE,), (CONFIG_BUFFER_SIZE,), (RELU6_GAMMA_BUFFER_SIZE,), (RELU6_BETA_BUFFER_SIZE,)], dtypes=[DepthConvData0Type, ConfigInst, ReluData0Type, ConfigInst, CinLoadData0Type, CinLoadData0Type], name="relu_bn")
    def relu6(cin, config_in, cout, config_out, gamma_conv, beta_conv):
        """
            ReLU Module: bias + batch norm + relu/relu6
            cin: input data
            config_in: input instruction 192-bit x5
            config_out: output instruction passed to downstream
            gamma_conv: multiplication parameter, if there is no batchnorm then it's normal bias: gamma=1
            beta_conv: addition parameter, if there is no batchnorm then it's normal bias: beta = bias
        """
        # write config_out
        config_out = hcl.compute((1,), lambda *args : config_in[args])
        # decode
        inst0, inst1, inst2, inst3, inst4 = decode_instructions(config_in)
        # set up signals
        CONV_EN             = hcl.compute((1,), lambda _ : inst3.layer_en[2], dtype=hcl.UInt(1))
        RELU_EN             = hcl.compute((1,), lambda _ : inst3.layer_en[3], dtype=hcl.UInt(1))
        RELU6_EN            = hcl.compute((1,), lambda _ : inst3.layer_en[4], dtype=hcl.UInt(1))
        BIAS_EN             = hcl.compute((1,), lambda _ : inst3.layer_en[7], dtype=hcl.UInt(1))
        BATCH_NORM_EN       = hcl.compute((1,), lambda _ : inst3.layer_en[10], dtype=hcl.UInt(1))
        BATCH_NORM_EN_DEPTH = hcl.compute((1,), lambda _ : inst3.layer_en[12], dtype=hcl.UInt(1))
        LAYER_IN_NUM_HW     = inst0.in_num_hw.asnode()
        LAYER_OUT_NUM_HW    = inst0.out_num_hw.asnode()
        LAYER_IN_H_HW       = inst0.in_h_hw.asnode()
        LAYER_IN_W_HW       = inst0.in_w_hw.asnode()
        LAYER_IN_NUM_T      = inst3.in_num_t.asnode()
        LAYER_OUT_NUM_T     = inst3.out_num_t.asnode()
        LAYER_IN_H_T        = inst3.in_h_t.asnode()
        LAYER_IN_W_T        = inst3.in_w_t.asnode()
        STRIDE              = inst2.stride.asnode()

        # a = LAYER_IN_NUM_T / RELU_LANE
        """
            type(LAYER_IN_NUM_T) is hcl.Scalar
            arg = convert_to_node(LAYER_IN_NUM_T) 
            type(arg) is hcl.TensorSlice
            We want arg.handle, but it says 
            *** heterocl.debug.TensorError: [Tensor] Cannot access attribute if type is not struct
            to solve this, you have to convert 'value' to 'node'.
            either use hcl.compute to get a new tensor, or use .asnode()
        """

        # branch conditions
        cond    = hcl.or_(RELU_EN, RELU6_EN, BIAS_EN, BATCH_NORM_EN)
        bias_en = hcl.and_(BIAS_EN, CONV_EN)

        # do computation
        # note: we only allow batch=1
        with hcl.for_(0, LAYER_OUT_NUM_HW / LAYER_OUT_NUM_T) as out_num_iter:
            with hcl.for_(0, LAYER_IN_W_HW / LAYER_IN_W_T)      as in_w_iter:
                with hcl.for_(0, LAYER_IN_H_HW / LAYER_IN_H_T)      as in_h_iter:
                    with hcl.for_(0, LAYER_IN_NUM_HW / LAYER_IN_NUM_T) as in_num_iter:
                        """ for each tile """
                        with hcl.for_(0, LAYER_IN_NUM_T / RELU_LANE) as o:
                            with hcl.for_(0, LAYER_IN_H_T / STRIDE)   as h : 
                                with hcl.for_(0, LAYER_IN_W_T / STRIDE) as w:
                                    # calculate index for beta and gamma
                                    beta_gamma_idx = in_num_iter * (LAYER_IN_NUM_T / RELU_LANE) + o
                                    # calculate index for feature map
                                    feat_idx =  (in_num_iter + in_h_iter * LAYER_IN_NUM_HW / LAYER_IN_NUM_T + in_w_iter * LAYER_IN_H_HW / LAYER_IN_H_T * LAYER_IN_NUM_HW / LAYER_IN_NUM_T) * (LAYER_IN_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T) \
                                        + (w + h * LAYER_IN_W_T + o * LAYER_IN_W_T * LAYER_IN_H_T)
                                    # fetch cin
                                    beta_SIMD = hcl.compute((1,), lambda *_ : beta_conv[beta_gamma_idx])
                                    gamma_SIMD = hcl.compute((1,), lambda *_ : gamma_conv[beta_gamma_idx])
                                    input_SIMD = hcl.compute((1,), lambda *_ : cin[feat_idx])
                                    # unpack
                                    beta_uint = hcl.unpack(beta_SIMD, dtype=hcl.UInt(32))
                                    gamma_uint = hcl.unpack(gamma_SIMD, dtype=hcl.UInt(32))
                                    cin_uint = hcl.unpack(input_SIMD, dtype=hcl.UInt(32))
                                    # bitcast
                                    beta_float = hcl.bitcast(beta_uint, hcl.Float(32))
                                    gamma_float = hcl.bitcast(gamma_uint, hcl.Float(32))
                                    cin_float = hcl.bitcast(cin_uint, hcl.Float(32))
                                    # calculate
                                    with hcl.if_(BATCH_NORM_EN_DEPTH):
                                        cout_float = hcl.compute((RELU_LANE,), lambda *args : gamma_float[args] * cin[args] + beta_float[args])
                                    with hcl.if_(RELU6_EN):
                                        cout_tmp = hcl.compute((RELU_LANE,), lambda *args : hcl.select(cin_float[args] < 0, hcl.cast(cin_float.dtype, 0), cin_float[args]))
                                        cout_float = hcl.compute((RELU_LANE,), lambda *args : hcl.select(cout_tmp[args] > 6, hcl.cast(cout_tmp.dtype, 6), cout_tmp[args]))
                                    with hcl.elif_(RELU_EN):
                                        cout_float = hcl.compute((RELU_LANE,), lambda *args : hcl.select(cin_float[args] < 0, hcl.cast(cin_float.dtype, 0), cin_float[args]))
                                    # bitcast back
                                    cout_uint = hcl.bitcast(cout_float, hcl.UInt(32))
                                    # pack
                                    cout_SIMD = hcl.pack(cout_uint, dtype=ReluData0Type)
                                    # write out packed result
                                    with hcl.if_(cond):
                                        cout[feat_idx] = cout_SIMD
                                    with hcl.else_():
                                        cout[feat_idx] = input_SIMD

    @def_([(RELU_CIN_BUFFER_SIZE,), (CONFIG_BUFFER_SIZE,), (RELU_COUT_BUFFER_SIZE,), (CONFIG_BUFFER_SIZE,), (RELU_GAMMA_BUFFER_SIZE,), (RELU_BETA_BUFFER_SIZE,)], dtypes=[DepthConvData0Type, ConfigInst, ReluData0Type, ConfigInst, CinLoadData0Type, CinLoadData0Type], name="relu_bn")
    def relu(cin, config_in, cout, config_out, gamma_conv, beta_conv):
        """
            ReLU Module: bias + batch norm + relu/relu6
            cin: input data
            config_in: input instruction 192-bit x5
            config_out: output instruction passed to downstream
            gamma_conv: multiplication parameter, if there is no batchnorm then it's normal bias: gamma=1
            beta_conv: addition parameter, if there is no batchnorm then it's normal bias: beta = bias
        """
        # write config_out
        config_out = hcl.compute((1,), lambda *args : config_in[args])
        # decode
        inst0, inst1, inst2, inst3, inst4 = decode_instructions(config_in)
        # set up signals
        CONV_EN             = hcl.compute((1,), lambda _ : inst3.layer_en[2], dtype=hcl.UInt(1))
        RELU_EN             = hcl.compute((1,), lambda _ : inst3.layer_en[3], dtype=hcl.UInt(1))
        RELU6_EN            = hcl.compute((1,), lambda _ : inst3.layer_en[4], dtype=hcl.UInt(1))
        BIAS_EN             = hcl.compute((1,), lambda _ : inst3.layer_en[7], dtype=hcl.UInt(1))
        BATCH_NORM_EN       = hcl.compute((1,), lambda _ : inst3.layer_en[10], dtype=hcl.UInt(1))
        BATCH_NORM_EN_DEPTH = hcl.compute((1,), lambda _ : inst3.layer_en[12], dtype=hcl.UInt(1))
        LAYER_IN_NUM_HW     = inst0.in_num_hw.asnode()
        LAYER_OUT_NUM_HW    = inst0.out_num_hw.asnode()
        LAYER_IN_H_HW       = inst0.in_h_hw.asnode()
        LAYER_IN_W_HW       = inst0.in_w_hw.asnode()
        LAYER_IN_NUM_T      = inst3.in_num_t.asnode()
        LAYER_OUT_NUM_T     = inst3.out_num_t.asnode()
        LAYER_IN_H_T        = inst3.in_h_t.asnode()
        LAYER_IN_W_T        = inst3.in_w_t.asnode()
        STRIDE              = inst2.stride.asnode()

        # a = LAYER_IN_NUM_T / RELU_LANE
        """
            type(LAYER_IN_NUM_T) is hcl.Scalar
            arg = convert_to_node(LAYER_IN_NUM_T) 
            type(arg) is hcl.TensorSlice
            We want arg.handle, but it says 
            *** heterocl.debug.TensorError: [Tensor] Cannot access attribute if type is not struct
            to solve this, you have to convert 'value' to 'node'.
            either use hcl.compute to get a new tensor, or use .asnode()
        """

        # branch conditions
        cond    = hcl.or_(RELU_EN, RELU6_EN, BIAS_EN, BATCH_NORM_EN)
        bias_en = hcl.and_(BIAS_EN, CONV_EN)

        # do computation
        # note: we only allow batch=1
        with hcl.for_(0, LAYER_OUT_NUM_HW / LAYER_OUT_NUM_T) as out_num_iter:
            with hcl.for_(0, LAYER_IN_W_HW / LAYER_IN_W_T)      as in_w_iter:
                with hcl.for_(0, LAYER_IN_H_HW / LAYER_IN_H_T)      as in_h_iter:
                    with hcl.for_(0, LAYER_IN_NUM_HW / LAYER_IN_NUM_T) as in_num_iter:
                        """ for each tile """
                        with hcl.for_(0, LAYER_IN_NUM_T / RELU_LANE) as o:
                            with hcl.for_(0, LAYER_IN_H_T / STRIDE)   as h : 
                                with hcl.for_(0, LAYER_IN_W_T / STRIDE) as w:
                                    # calculate index for beta and gamma
                                    beta_gamma_idx = in_num_iter * (LAYER_IN_NUM_T / RELU_LANE) + o
                                    # calculate index for feature map
                                    feat_idx =  (in_num_iter + in_h_iter * LAYER_IN_NUM_HW / LAYER_IN_NUM_T + in_w_iter * LAYER_IN_H_HW / LAYER_IN_H_T * LAYER_IN_NUM_HW / LAYER_IN_NUM_T) * (LAYER_IN_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T) \
                                        + (w + h * LAYER_IN_W_T + o * LAYER_IN_W_T * LAYER_IN_H_T)
                                    # fetch cin
                                    beta_SIMD = hcl.compute((1,), lambda *_ : beta_conv[beta_gamma_idx])
                                    gamma_SIMD = hcl.compute((1,), lambda *_ : gamma_conv[beta_gamma_idx])
                                    input_SIMD = hcl.compute((1,), lambda *_ : cin[feat_idx])
                                    # unpack
                                    beta_uint = hcl.unpack(beta_SIMD, dtype=hcl.UInt(32))
                                    gamma_uint = hcl.unpack(gamma_SIMD, dtype=hcl.UInt(32))
                                    cin_uint = hcl.unpack(input_SIMD, dtype=hcl.UInt(32))
                                    # bitcast
                                    beta_float = hcl.bitcast(beta_uint, hcl.Float(32))
                                    gamma_float = hcl.bitcast(gamma_uint, hcl.Float(32))
                                    cin_float = hcl.bitcast(cin_uint, hcl.Float(32))
                                    # calculate
                                    with hcl.if_(BATCH_NORM_EN_DEPTH):
                                        cout_float = hcl.compute((RELU_LANE,), lambda *args : gamma_float[args] * cin[args] + beta_float[args])
                                    with hcl.if_(RELU6_EN):
                                        cout_tmp = hcl.compute((RELU_LANE,), lambda *args : hcl.select(cin_float[args] < 0, hcl.cast(cin_float.dtype, 0), cin_float[args]))
                                        cout_float = hcl.compute((RELU_LANE,), lambda *args : hcl.select(cout_tmp[args] > 6, hcl.cast(cout_tmp.dtype, 6), cout_tmp[args]))
                                    with hcl.elif_(RELU_EN):
                                        cout_float = hcl.compute((RELU_LANE,), lambda *args : hcl.select(cin_float[args] < 0, hcl.cast(cin_float.dtype, 0), cin_float[args]))
                                    # bitcast back
                                    cout_uint = hcl.bitcast(cout_float, hcl.UInt(32))
                                    # pack
                                    cout_SIMD = hcl.pack(cout_uint, dtype=ReluData0Type)
                                    # write out packed result
                                    with hcl.if_(cond):
                                        cout[feat_idx] = cout_SIMD
                                    with hcl.else_():
                                        cout[feat_idx] = input_SIMD



    # bilinear upsample
    @def_([(UP_CIN_BUFFER_SIZE,),(CONFIG_BUFFER_SIZE,),(UP_COUT_BUFFER_SIZE,),(CONFIG_BUFFER_SIZE,)], dtypes=[ReluData0Type, ConfigInst, UpsampleData0Type, ConfigInst], name="bilinear_upsample")
    def upsample(cin, config_in, cout, config_out):
        # write config_out
        config_out = hcl.compute((1,), lambda x : config_in[x])
        # decode
        inst0, inst1, inst2, inst3, inst4 = decode_instructions(config_in)
        # set up control signals
        UPSAMPLE_EN         = hcl.compute((1,), lambda _ : inst3.layer_en[6], dtype=UInt(1), name='UPSAMPLE_EN')
        LAYER_IN_NUM_HW     = inst0.in_num_hw.asnode()
        LAYER_OUT_NUM_HW    = inst0.out_num_hw.asnode()
        LAYER_IN_H_HW       = inst0.in_h_hw.asnode()
        LAYER_IN_W_HW       = inst0.in_w_hw.asnode()
        LAYER_IN_NUM_T      = inst3.in_num_t.asnode()
        LAYER_OUT_NUM_T     = inst3.out_num_t.asnode()
        LAYER_IN_H_T        = inst3.in_h_t.asnode()
        LAYER_IN_W_T        = inst3.in_w_t.asnode()
        STRIDE              = inst2.stride.asnode()

        line_buff = hcl.compute((MAX_TILE_WIDTH * 2,), lambda x : 0, dtype=ReluData0Type, name='line_buff')

        with hcl.for_(0, LAYER_OUT_NUM_HW / LAYER_OUT_NUM_T) as out_num_iter:
            with hcl.for_(0, LAYER_IN_W_HW / LAYER_IN_W_T)      as in_w_iter:
                with hcl.for_(0, LAYER_IN_H_HW / LAYER_IN_H_T)      as in_h_iter:
                    with hcl.for_(0, LAYER_IN_NUM_HW / LAYER_IN_NUM_T) as in_num_iter:
                        """ for each tile """
                        with hcl.for_(0, LAYER_IN_NUM_T / RELU_LANE) as o:
                            with hcl.for_(0, LAYER_IN_H_T / STRIDE)   as h : 
                                with hcl.for_(0, LAYER_IN_W_T / STRIDE) as w:
                                    # load two lines of input feature map
                                    feat_idx =  (in_num_iter + in_h_iter * LAYER_IN_NUM_HW / LAYER_IN_NUM_T + in_w_iter * LAYER_IN_H_HW / LAYER_IN_H_T * LAYER_IN_NUM_HW / LAYER_IN_NUM_T) * (LAYER_IN_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T) \
                                        + (w + h * LAYER_IN_W_T + o * LAYER_IN_W_T * LAYER_IN_H_T)
                                    hcl.update(line_buff, lambda x : cin[x + feat_idx])
                                    # fetch SIMD data
                                    # TODO: this API can be simplified maybe
                                    a_SIMD = hcl.compute((1,), lambda _ : line_buff[h], dtype=ReluData0Type)
                                    b_SIMD = hcl.compute((1,), lambda _ : line_buff[h+1], dtype=ReluData0Type)
                                    c_SIMD = hcl.compute((1,), lambda _ : line_buff[h+w], dtype=ReluData0Type)
                                    d_SIMD = hcl.compute((1,), lambda _ : line_buff[h+w+1], dtype=ReluData0Type)

                                    # a_SIMD = line_buff[h].asnode() this doesn't work

                                    # unpack
                                    a_uint = hcl.unpack(a_SIMD, dtype=hcl.UInt(32))
                                    b_uint = hcl.unpack(b_SIMD, dtype=hcl.UInt(32))
                                    c_uint = hcl.unpack(c_SIMD, dtype=hcl.UInt(32))
                                    d_uint = hcl.unpack(d_SIMD, dtype=hcl.UInt(32))
                                    # bitcast
                                    a_float = hcl.bitcast(a_uint, hcl.Float(32))
                                    b_float = hcl.bitcast(b_uint, hcl.Float(32))
                                    c_float = hcl.bitcast(c_uint, hcl.Float(32))
                                    d_float = hcl.bitcast(d_uint, hcl.Float(32))
                                    # calculate interpolated points
                                    out01_float = hcl.compute(a_float.shape, lambda x : (a_float[x] + b_float[x]) / 2, dtype=hcl.Float(32), name='out_01')
                                    out10_float = hcl.compute(a_float.shape, lambda x : (a_float[x] + c_float[x]) / 2, dtype=hcl.Float(32), name='out_10')
                                    out11_float = hcl.compute(a_float.shape, lambda x : (a_float[x] + b_float[x] + c_float[x] + d_float[x]) /4, dtype=hcl.Float(32), name='out_11') 
                                    out21_float = hcl.compute(a_float.shape, lambda x : (c_float[x] + d_float[x]) / 2, dtype=hcl.Float(32), name='out_21')
                                    # bitcast back
                                    out01_uint = hcl.bitcast(out01_float, hcl.UInt(32))
                                    out10_uint = hcl.bitcast(out10_float, hcl.UInt(32))
                                    out11_uint = hcl.bitcast(out11_float, hcl.UInt(32))
                                    out21_uint = hcl.bitcast(out21_float, hcl.UInt(32))
                                    # pack
                                    # TODO: bug? stmt_stack size of out01_SIMD is 0.
                                    out01_SIMD = hcl.pack(out01_uint, dtype=UpsampleData0Type, name='out01_SIMD')
                                    out10_SIMD = hcl.pack(out10_uint, dtype=UpsampleData0Type, name='out10_SIMD')
                                    out11_SIMD = hcl.pack(out11_uint, dtype=UpsampleData0Type, name='out11_SIMD')
                                    out21_SIMD = hcl.pack(out21_uint, dtype=UpsampleData0Type, name='out21_SIMD')
                                    # write out
                                    with hcl.if_(UPSAMPLE_EN):
                                        cout[feat_idx] = a_SIMD
                                        cout[feat_idx + 1] = out01_SIMD
                                        # next row
                                        cout[feat_idx * 2 + LAYER_IN_W_T * 2] = out10_SIMD
                                        cout[feat_idx * 2 + LAYER_IN_W_T * 2 + 1] = out11_SIMD
                                        # next row
                                        cout[feat_idx * 2 + LAYER_IN_W_T * 4] = c_SIMD
                                        cout[feat_idx * 2 + LAYER_IN_W_T * 4 + 1] = out21_SIMD
                                    with hcl.else_():
                                        cout[feat_idx] = a_SIMD


    # not availabel in hlib.nn
    @def_([(ADD_CIN1_BUFFER_SZE,), (ADD_CIN2_BUFFER_SIZE,), (CONFIG_BUFFER_SIZE,), (ADD_COUT_BUFFER_SIZE,), (CONFIG_BUFFER_SIZE,)], dtypes=[CinLoadData0Type, ReluData0Type, ConfigInst, ReluData0Type, ConfigInst], name="add")
    def add(cin1, cin2, config_in, cout, config_out):
        pass
                

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
        cin_load_0          = hcl.compute((DEPTH_CIN_BUFFER_SIZE,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_cin_load_0")
        weight_load_0       = hcl.compute((DEPTH_WEIGHT_BUFFER_SIZE,), lambda *_ : 0, dtype=WeightLoadData0Type, name="fifo_weight_load_0")
        weight_load_1       = hcl.compute((CONV_WEIGHT_BUFFER_SIZE,), lambda *_ : 0, dtype=WeightLoadData0Type, name="fifo_weight_load_1")
        depth_conv_0        = hcl.compute((DEPTH_CIN_BUFFER_SIZE,), lambda *_ : 0, dtype=DepthConvData0Type, name="fifo_depth_conv_0")
        relu6_0             = hcl.compute((RELU6_CIN_BUFFER_SIZE,), lambda *_ : 0, dtype=ReluData0Type, name="fifo_relu6_0")
        conv_0              = hcl.compute((CONV_CIN_BUFFER_SIZE,), lambda *_ : 0, dtype=DepthConvData0Type, name="fifo_conv_0")
        add_0               = hcl.compute((ADD_COUT_BUFFER_SIZE,), lambda *_ : 0, dtype=ReluData0Type, name="fifo_add_0")
        relu_0              = hcl.compute((ADD_CIN2_BUFFER_SIZE,), lambda *_ : 0, dtype=ReluData0Type, name="fifo_relu_0")
        upsample_0          = hcl.compute((UP_COUT_BUFFER_SIZE,), lambda *_ : 0, dtype=UpsampleData0Type, name="fifo_upsample_0")
        merge_0             = hcl.compute((UP_COUT_BUFFER_SIZE,), lambda *_ : 0, dtype=UpsampleData0Type, name="fifo_merge_0")
        cin_prev_0          = hcl.compute((ADD_CIN1_BUFFER_SZE,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_cin_prev_0")
        beta_depth          = hcl.compute((RELU6_BETA_BUFFER_SIZE,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_beta_depth")
        gamma_depth         = hcl.compute((RELU6_GAMMA_BUFFER_SIZE,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_gamma_depth")
        beta_conv           = hcl.compute((RELU_BETA_BUFFER_SIZE,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_beta_conv")
        gamma_conv          = hcl.compute((RELU_GAMMA_BUFFER_SIZE,), lambda *_ : 0, dtype=CinLoadData0Type, name="fifo_gamma_conv")
        config_prev_load    = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_prev_load")
        config_weight_load  = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_weight_load")
        config_depth_conv   = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_depth_conv")
        config_relu6        = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_relu6")
        config_conv         = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_conv")
        config_add          = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_add")
        config_relu         = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_relu")
        config_upsample     = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_upsample")
        config_merge        = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name='config_merge')
        config_data_write   = hcl.compute((CONFIG_BUFFER_SIZE,), lambda *_ : 0, dtype=ConfigInst, name="config_data_write")

        # connect modules 
        cin_load(global_cin, layer_config, cin_load_0, config_prev_load)
        cin_load_prev(global_prev_cin, config_prev_load, cin_prev_0, config_weight_load)
        weight_load(global_weight, global_bias, config_weight_load, weight_load_0, weight_load_1, beta_depth, gamma_depth, beta_conv, gamma_conv, config_depth_conv)
        depth_conv(cin_load_0, weight_load_0, config_depth_conv, depth_conv_0, config_relu6)
        relu6(depth_conv_0, config_relu6, relu6_0, config_conv, beta_depth, gamma_depth)
        conv(relu6_0, weight_load_1, config_conv, conv_0, config_relu)
        relu(conv_0, config_relu, relu_0, config_add, beta_conv, gamma_conv)
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
    global_cin = hcl.placeholder((GLOBAL_CIN_SIZE,), name="global_cin", dtype=bus_t0)
    global_prev_cin = hcl.placeholder(
        (GLOBAL_PREV_CIN_SIZE,), name="global_prev_cin", dtype=bus_t0
    )
    global_weight = hcl.placeholder(
        (GLOBAL_WEIGHT_SIZE,), name="global_weight", dtype=bus_t1
    )
    global_bias = hcl.placeholder((GLOBAL_BIAS_SIZE,), name="global_bias", dtype=bus_t2)
    global_cout = hcl.placeholder(
        (GLOBAL_COUT_SIZE,), name="global_cout", dtype=bus_t0
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
    print(hcl.lower(s))
    p = hcl.platform.aws_f1
    p.config(compile='vitis', mode='debug')
    # p = "vhls"

    code = str(hcl.build(s, p, name="main"))
    with open("flexcnn.cpp", "w") as f:
        f.write(code)


if __name__ == "__main__":
    test_flexcnn()