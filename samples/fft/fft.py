import numpy as np
import heterocl as hcl

hcl.init(hcl.Float())

def PreCompute_BitReverse_Pattern(L):
    """
    Parameters
    ----------
    L : int
        Length of the input vector.

    Returns
    -------
    IndexTable : vector of int
        Permutation pattern.
    """
    bit_width = int(np.log2(L))
    IndexTable = np.zeros((L), dtype='int')
    for i in range(L):
        b = '{:0{width}b}'.format(i, width=bit_width)
        IndexTable[i] = int(b[::-1], 2)
    return IndexTable

L = 1024

def top(target=None):

    def fft(X_real, X_imag, IndexTable, F_real, F_imag):
        L = X_real.shape[0]
        if np.log2(L) % 1 > 0:
            raise ValueError("Length of input vector (1d tensor) must be power of 2")
        num_stages = int(np.log2(L))

        # bit reverse permutation
        hcl.update(F_real, lambda i: X_real[IndexTable[i]], name='F_real_update')
        hcl.update(F_imag, lambda i: X_imag[IndexTable[i]], name='F_imag_update')

        with hcl.Stage("Out"):
            one = hcl.scalar(1, dtype="int32")
            with hcl.for_(0, num_stages) as stage:
                DFTpts = one[0] << (stage + 1)
                numBF = DFTpts / 2
                e = -2 * np.pi / DFTpts
                a = hcl.scalar(0)
                with hcl.for_(0, numBF) as j:
                    c = hcl.scalar(hcl.cos(a[0]))
                    s = hcl.scalar(hcl.sin(a[0]))
                    a[0] = a[0] + e
                    with hcl.for_(j, L + DFTpts - 1, DFTpts) as i:
                        i_lower = i + numBF
                        temp_r = hcl.scalar(F_real[i_lower] * c - F_imag[i_lower] * s)
                        temp_i = hcl.scalar(F_imag[i_lower] * c + F_real[i_lower] * s)
                        F_real[i_lower] = F_real[i] - temp_r[0]
                        F_imag[i_lower] = F_imag[i] - temp_i[0]
                        F_real[i] = F_real[i] + temp_r[0]
                        F_imag[i] = F_imag[i] + temp_i[0]

    X_real = hcl.placeholder((L,))
    X_imag = hcl.placeholder((L,))
    IndexTable = hcl.placeholder((L,), dtype="int32")
    F_real = hcl.placeholder((L,))
    F_imag = hcl.placeholder((L,))

    s = hcl.create_schedule([X_real, X_imag, IndexTable, F_real, F_imag], fft)
    return hcl.build(s, target=target)

f = top()

x_real_np = np.random.random((L))
x_imag_np = np.random.random((L))
x_np = x_real_np + 1j * x_imag_np

out_np = np.fft.fft(x_np)
out_real_np = out_np.real
out_imag_np = out_np.imag

x_real_hcl = hcl.asarray(x_real_np)
x_imag_hcl = hcl.asarray(x_imag_np)
index_table_hcl = hcl.asarray(PreCompute_BitReverse_Pattern(L), dtype="int")

out_real_hcl = hcl.asarray(np.zeros((L)))
out_imag_hcl = hcl.asarray(np.zeros((L)))

f(x_real_hcl, x_imag_hcl, index_table_hcl, out_real_hcl, out_imag_hcl)

np.testing.assert_allclose(out_real_np, out_real_hcl.asnumpy(), rtol=1e-02, atol=1e-3)
np.testing.assert_allclose(out_imag_np, out_imag_hcl.asnumpy(), rtol=1e-02, atol=1e-3)
