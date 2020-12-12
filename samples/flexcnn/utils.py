"""
utility functions for FlexCNN
"""

import heterocl as hcl

KERNEL_SIZE_MAX = 5


def depth_calc_inout_idx(o_idx, h_idx, w_idx, h_idx_t, w_idx_t, c_idx_t, h_t, w_t, c_t, h_hw, w_hw, c_hw, cout_hw, kernel_size, stride):
    """ 
        calculate feature map buffer index

        o_idx: convolution channel index in tile (must be divided by SIMD_LANE)
        h_idx: convolution center h_index in tile
        w_idx: convolution center w_index in tile
        h_idx_t: h_index of current tile
        w_idx_t: w_index of current tile
        c_idx_t: c_index of current tile
        h_t:   height of the tile
        w_t:   width of the tile
        c_t:   channel number of the tile
        h_hw:  hardware complete input feature map height 
        w_hw:  hardware complete input feature map width
        c_hw:  hardware complete input feature map channel number
        kernel_size: kernel size
        stride: stride

        return: hcl.Tensor
        in_indices: a list of integers, the indices of input feature for one convolution (SIMD packed)
        e.g. for 3x3 kernel, len(return) = 9
        out_index: output index
    """

    in_indices = hcl.compute((KERNEL_SIZE_MAX * KERNEL_SIZE_MAX, ), lambda *_ : 0, name="depth_input_indices") 

    tile_num_h = h_hw / h_t
    tile_num_w = w_hw / w_t
    tile_num_c = c_hw / c_t    
    passed_tile_num = w_idx_t * tile_num_c * tile_num_h + h_idx_t * tile_num_c + c_idx_t
    tile_size = h_t * w_t * c_t
    base = tile_size * passed_tile_num

    with hcl.for_(0, kernel_size) as k1:
        with hcl.for_(0, kernel_size) as k2:
            tmp_h = h_idx - (kernel_size - 1) / 2 + k1
            tmp_w = w_idx - (kernel_size - 1) / 2 + k2
            in_indices[k1 * kernel_size + k2] = base + o_idx * h_t * w_t + tmp_h * w_t + tmp_w

    # calculate output index
    out_tile_size = (h_t/stride) * (w_t/stride) * cout_hw
    out_index = passed_tile_num * out_tile_size + (h_idx/stride) * (w_t/stride) + w_idx/stride

    return in_indices, out_index


def depth_calc_weights_idx(cout_idx, kernel_size, channel_in, channel_index):
    """
        calculate weights buffer index

        cout_idx: output channel index, which is also the index of weight kernel
        kernel_size: the size of kernel
        channel_in: the number of input channels (divided by SIMD)
        index: channel index (SIMD packet index) 

        return: hcl.Tensor
        a list of integers, the indices of weight parameters for one convolution (SIMD packed)
        e.g. for 3x3 kernel, len(return) = 9
    """

    indices = hcl.compute((KERNEL_SIZE_MAX * KERNEL_SIZE_MAX,), lambda *_ : 0, name="depth_weights_indices")

    kernel_elem_count = kernel_size * kernel_size * channel_in
    base = cout_idx * kernel_elem_count + channel_index * kernel_size * kernel_size

    with hcl.for_(0, kernel_size) as k1:
        with hcl.for_(0, kernel_size) as k2:
            indices[k1 * kernel_size + k2] = base + k1 * kernel_size + k2

    return indices




def depth_calc_output_idx(in_h_hw, in_w_hw, ):
    """
        calculate output index for current convolution
        
        return: hcl.Tensor
        an integer, the index of output feature for one convolution (SIMD packed)
    """
    
    return [0]