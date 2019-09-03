import heterocl as hcl
import numpy as np
from gemm_main import *

#dtypes = [hcl.Int(32), hcl.Float(), hcl.Fixed(32, 16)]
#for dtype in dtypes:
#time_gemm(hcl.Int(32), 10, 10, 10, 'sdaccel_sw_emu')
time_gemm(hcl.Int(32), 10, 10, 10, 'sdaccel_sw_emu')
