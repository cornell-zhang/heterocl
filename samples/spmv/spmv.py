import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import random
from scipy.stats import rv_continuous
import heterocl as hcl

class CustomDistribution(rv_continuous):
    def _rvs(self, size=None, random_state=None):
        return random_state.randn(*size)

dimX, dimY = 1024, 1024
X = CustomDistribution(seed=2906)
Y = X()  # get a frozen version of the distribution
S = random(dimX, dimY, density=0.25, random_state=2906, data_rvs=Y.rvs)
sA = sparse.csr_matrix(S.A)

value = sA.data
column_index = sA.indices
row_pointers = sA.indptr

def spmv(stream=False):
    dtype = hcl.Float()
    hcl.init()

    col = hcl.placeholder(column_index.shape, name="col")
    row = hcl.placeholder(row_pointers.shape, "row_ptr")
    val = hcl.placeholder(value.shape, dtype=dtype, name="value")
    vector = hcl.placeholder((1024,), dtype=dtype, name="vector")
    out = hcl.placeholder((1024,), dtype=dtype, name="out")

    def kernel(col, row, val, vector, out):
        def loop(i):
            with hcl.Stage("A"):
                tmp = hcl.scalar(0, "tmp")
                s = hcl.scalar(row[i], "s")
                e = hcl.scalar(row[i+1], "e")

            with hcl.Stage("B"):
                with hcl.for_(s.v, e.v, name="c") as c:
                    cid = hcl.scalar(col[c], "cid")
                    tmp.v += val[c] * vector[cid.v]

            with hcl.Stage("C"):
                out[i] = tmp.v

        hcl.mutate((1024,), lambda i: loop(i), "main")
    
    s = hcl.create_schedule([col, row, val, vector, out], kernel)
    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim")
    s.to([col, row, val, vector], p.xcel)
    s.to(kernel.main.out, p.host)

    if stream:
        # duplicate the inner loop with variable latency
        s.duplicate(kernel.main.B, factor=2)
        # create FIFOs between stages inside the pipeline loop
        s.to(kernel.main.B.tmp, kernel.main.C)
        s.to([kernel.main.B.s, kernel.main.B.e], kernel.main.B)

    f = hcl.build(s, target=p)
    values = [column_index, row_pointers, value]

    # input vector and output res
    values.append(np.random.uniform(low=0, high=10.0, size=(1024,)))
    values.append(np.random.uniform(low=0, high=10.0, size=(1024,)))

    args = hcl.util.gen_hcl_array(s, values)
    f.inspect(args)

if __name__ == "__main__":
    spmv()




