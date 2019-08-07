""" author: Guyue Huang (gh424@cornell.edu)
ppac-gemm data generater
"""
import numpy as np
import numpy.random as rd
from ppac_common import compact, bvec2x

m, n, k 		= 4, 4, 64
v_bits, m_bits 	= 1, 1
xLen 			= 64

mat_A = rd.randint(2**m_bits, size=(m, k))
mat_B = rd.randint(2**v_bits, size=(k, n))
mat_C = np.dot(mat_A, mat_B)
golden = list(mat_C.flatten('F'))   # column major

data_A = np.zeros((m, xLen))
data_B = np.zeros((n, xLen))
for i in range(m):
    data_A[i,:] = compact(mat_A[i,:], k, m_bits, xLen)
for j in range(n):
    data_B[j,:] = compact(mat_B[:,j], k, v_bits, xLen)

word_A = [bvec2x(vec) for vec in list(data_A)]
word_B = [bvec2x(vec) for vec in list(data_B)]

with open('csrcmacro.txt', 'w') as of:
    of.write('#define M '+str(m)+'\n')
    of.write('#define N ' + str(n) + '\n')
    of.write('#define K ' + str(k) + '\n')
    of.write('#define DATAA '+'0x'+word_A[0])
    for s in word_A[1:]:
        of.write(',\\\n'+'\t0x'+s)
    of.write('\n\n')
    of.write('#define DATAB '+'\t0x'+word_B[0])
    for s in word_B[1:]:
        of.write(',\\\n'+'\t0x'+s)
    of.write('\n\n')
    of.write('#define DATAGOLD ')
    for n in golden[:-1]:
        of.write(str(n)+', ')
    of.write(str(golden[-1])+'\n')
of.close()

with open('bareMdata.txt','w') as of:
    of.write('data_A:\n')
    for s in word_A:
        of.write('\t.dword '+s+'\n')
    of.write('\ndata_B:\n')
    for s in word_B:
        of.write('\t.dword '+s+'\n')
    of.write('\ndata_C:\n')
    for s in range(m*n):
        of.write('\t.dword 0x0\n')
of.close()

with open('goldennumber.txt','w') as of:
    for n in golden:
        of.write(str(n)+'\n')
