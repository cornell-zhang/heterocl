import heterocl as hcl
import numpy as np

def test (DRAM):
        hcl.print ((), f"DRAM: {DRAM.shape} {DRAM.dtype}\n")
        hcl.print(DRAM)

DRAM    = hcl.placeholder((4,2), "dram", dtype=hcl.UInt(64))
s = hcl.create_schedule([DRAM], test)
f = hcl.build(s)
dram    = hcl.asarray (np.zeros(DRAM.shape), dtype=DRAM.dtype)
f(dram)
