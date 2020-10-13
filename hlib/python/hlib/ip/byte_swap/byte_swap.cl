    
    
    void byte_swap(int* input, int* vec, int len)
    for (int k = 0; k < len; k++) {{
      vec[k] = my_byteswap(input[k]);
    }}