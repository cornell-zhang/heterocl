"""author: Guyue Huang (gh424@cornell.edu)
ppac-gemm c code gen
"""
from ppac_common import *

def getCSrc(data_fname, golden_fname, head_fname, o_fname, dims, bits=1, xLen=64):

    m, n, k = dims

    def printKernel(of):

        of.write('//save data_A\n')
        for i in range(m):
            of.write(doLoad(('data_A+%d'%(8*i)), i))

        of.write('\n//do MVP\n')
        for j in range(n):
            of.write('\tlw a1, data_B+%d\n'%(8*j))
            #TODO: bit-mask
            of.write(doMVP(funct='72'))     #'1001000'
            for i in range(m):
                of.write(doStore(('data_C+%d'%(4*(j*m+i))),i))

    def printTest(of, golden):
        for n, gnum in enumerate(golden):
            of.write('\tlw a0, data_C+%d\n'%(4*n))
            of.write(('\tTEST_CASE(%d, a0, '%(n+1+1)) + hex(int(gnum)) + ', )\n')
        of.write('\n\tTEST_PASSFAIL\n')

    with open(o_fname, 'w') as of:

        with open(head_fname, 'r') as hf:
            of.write(hf.read())
        hf.close()

        of.write('\tRVTEST_WITH_ROCC\n')
        of.write('start:\n\tRVTEST_CODE_BEGIN\n')

        printKernel(of)
        of.write('\n\n')
        goldnum = []
        with open(golden_fname, 'r') as gf:
            goldnum += (gf.read()).split()
        if not len(goldnum) == m*n:
            raise Exception('golden number should have %d but %d'%(m*n, len(goldnum)))
        printTest(of, goldnum)

        of.write('\tRVTEST_CODE_END\n\n')

        with open(data_fname, 'r') as df:
            s = "\t.data\n\tRVTEST_DATA_BEGIN\n\n" + "\tTEST_DATA\n" + df.read()
        df.close()
        of.write(s)

        of.write('\tRVTEST_DATA_END\n')

if __name__== '__main__':
    getCSrc('bareMdata.txt', 'goldennumber.txt', 'headcode.txt', 'test.S',
            dims=[4, 4, 64])

    #TODO: headcode.txt ppac_common code definition

