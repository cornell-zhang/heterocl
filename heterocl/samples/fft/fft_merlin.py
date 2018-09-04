import numpy as np
import heterocl as hcl
from fft import PreCompute_BitReverse_Pattern, top, L
import os

L = 1024
X_real = hcl.placeholder((L,))
X_imag = hcl.placeholder((L,))
IndexTable = hcl.placeholder((L,), dtype="int32")

code = top('merlinc')
with open('fft_kernel.cpp', 'w') as f:
    f.write(code)

# Prepare input
data_file = open('input.dat', 'w')
x_real_np = np.random.random((L))
data_file.write('\t'.join([str(n) for n in x_real_np.tolist()]))
data_file.write('\n')
x_imag_np = np.random.random((L))
data_file.write('\t'.join([str(n) for n in x_imag_np.tolist()]))
data_file.write('\n')
x_np = x_real_np + 1j * x_imag_np
data_file.write('\t'.join([str(n) for n in PreCompute_BitReverse_Pattern(L).tolist()]))
data_file.close()

# Prepare reference
out_np = np.fft.fft(x_np)
out_real_np = out_np.real
out_imag_np = out_np.imag

# Here we use gcc to evaluate the functionality
os.system('g++ -std=c++11 fft_host.cpp fft_kernel.cpp')
os.system('./a.out input.dat')

# Read output
with open('output.dat', 'r') as f:
    out_real_hcl = [float(n) for n in f.readline().split('\t')]
    out_imag_hcl = [float(n) for n in f.readline().split('\t')]

    np.testing.assert_allclose(out_real_np, np.array(out_real_hcl),
            rtol=1e-03, atol=1e-4)
    np.testing.assert_allclose(out_imag_np, np.array(out_imag_hcl),
            rtol=1e-03, atol=1e-4)

os.system('rm a.out *.dat')
print "Success."
