import heterocl as hcl
import numpy as np
from smith_waterman_main import *

f = top("vhls_csim")

# add a very simple test
_seqA_np = np.ones((num, lenA))
for i in range(0, 4):
    _seqA_np[0][i] = 2
_seqB_np = np.ones((num, lenB))
_seqA = hcl.asarray(_seqA_np, dtype)
_seqB = hcl.asarray(_seqB_np, dtype)
_consensusA = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)
_consensusB = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)
f(_seqA, _seqB, _consensusA, _consensusB)
_consensusA_np = _consensusA.asnumpy()
_consensusB_np = _consensusB.asnumpy()
for i in range(0, 256):
    if i < 124:
        assert _consensusA_np[0][i] == 1
    else:
        assert _consensusA_np[0][i] == 0
