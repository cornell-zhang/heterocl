# codeing -*utf-8*-
import random
import numpy as np
import heterocl as hcl

# data type for encoding and channel
DTYPE = hcl.UInt(8)

M = 128 # array size
N = int(np.floor(np.log2(M)))

def top(target = None):
    input_arr = hcl.placeholder((M,))
    sort_arr  = hcl.placeholder((M,))
    size  = hcl.placeholder((1,), 'size')
    matrix = hcl.placeholder((M,2), 'matrix')

    # the sorting function offload to FPGA
    with hcl.Stage('S') as S:

        # sort the matrix with 2**n block
        def merge_sort(n):

            # compute the size of block to traverse each epochh
            block = hcl.compute((1,), lambda x: 1, 'block', DTYPE)
            with hcl.for_(0,n):
                block[0] = block[0] * 2
            trip = hcl.compute((1,), lambda x: 1 + M / block[0], 'trip')

            # sort two blocks with two-finger alogrithm
            with hcl.for_(0,trip[0]) as k:
                index_s = hcl.compute((1,), lambda x: k*block[0])
                index_e = hcl.compute((1,), lambda x: (k+1)*block[0])
                with hcl.if_(index_e[0] > M):
                    index_e[0] = M
                index_m = hcl.compute((1,), lambda x: (index_s[0] + index_e[0])/2)

                # ordinary sorting alogorithm
                #sort(index_s[0], index_e[0])

                # non-recursive merge sort
                merge(index_s[0], index_m[0], index_e[0])

        # merge (s, m-1) and (m, e-1)
        def merge(start, middle, end):
            f1 = hcl.compute((1,), lambda x: start)
            f2 = hcl.compute((1,), lambda x: middle)
            out = hcl.compute((M,), lambda x: 0)
            with hcl.for_(start,end) as index:
                with hcl.if_(hcl.or_(hcl.and_(f1[0] < middle, matrix[f1[0]][1] < matrix[f2[0]][1]), f2[0] == end)):
                    out[index] = matrix[f1[0]][1]
                    f1[0] = f1[0] + 1
                with hcl.else_():
                    out[index] = matrix[f2[0]][1]
                    f2[0] = f2[0] + 1

            # move the result back to matrix
            with hcl.for_(start,end) as idx:
                matrix[idx][1] = out[idx]

        def sort(start, end):
            index_min = hcl.compute((1,), lambda x:0)
            temp = hcl.compute((M,), lambda x: 0)
            with hcl.for_(start, end) as i:
                index_min[0], temp[i-start] = find_min(start, end)
                matrix[index_min[0]][1] = 10000
            with hcl.for_(start, end) as k:
                matrix[k][1] = temp[k-start]

        def find_min(start, end):
            index = hcl.compute((1,), lambda x: start)
            value = hcl.compute((1,), lambda x: matrix[start][1])
            with hcl.for_(start, end) as i:
                with hcl.if_(matrix[i][1] < value[0]):
                    index[0] = i
                    value[0] = matrix[i][1]
            return index[0], value[0]

        def init_matrix(x):
            matrix[x][0] = input_arr[x];
            matrix[x][1] = input_arr[x];
            matrix[x][2] = 0;

        # initialize the frequency array
        hcl.mut_compute((M,), lambda x: init_matrix(x), 'matrix')
        hcl.mut_compute((N+1,), lambda x: merge_sort(x), 'sort')

        # find first and second find, insert value and update index
        new_v = hcl.compute((1,), lambda x: 0, 'new_value')
        new_i = hcl.compute((1,), lambda x: 0, 'new_index')

    s = hcl.create_schedule([input_arr, sort_arr, size, matrix])

    print hcl.lower(s)

    # make scheme schedule
    return hcl.build(s, target = target)


# read data from train text
init_arr = [random.randint(1,200) for c in range(128)]
sort_arr = [0 for c in range(128)]

# build function and test encoding
f = top()

hcl_input  = hcl.asarray(np.array(init_arr), hcl.Int())
hcl_output = hcl.asarray(np.array(sort_arr), hcl.Int())
hcl_size   = hcl.asarray(np.array([N]), hcl.Int())
hcl_matrix = hcl.asarray(np.zeros((M,2)), hcl.Int())

f(hcl_input, hcl_output, hcl_size, hcl_matrix)
matrix = hcl_matrix.asnumpy()

print '\n[INFO] original and sorted array'
print np.array(matrix)
