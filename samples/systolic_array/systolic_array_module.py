import heterocl as hcl
import numpy as np

def systolic(m=2, k=2, n=2, w=2, h=2, dtype=hcl.Int(), target=None):

    hcl.init(dtype)

    Ain = hcl.placeholder((m, k), dtype=dtype, name="A")
    Bin = hcl.placeholder((k, n), dtype=dtype, name="B")
    # Output = hcl.placeholder((m, n), dtype=dtype, name="output")

    def systolic_array(A, B):

        # define modules with new decorator
        @hcl.def2_([(1,), (1,), (3,)], amount=w*h)
        def pe(a, b, out):
            out[0] = a
            out[1] = b
            out[2] = a * b

        # each k calls of update function calculate one block of result matrix
        # b_row: block row index
        # b_col: block col index
        def update(b_row, b_col, k, O):
            # fetch input
            localA = []
            localB = []
            for input_a in range(h):
                localA.append(hcl.compute((1,), lambda x : A[input_a + h * b_row, k], "localA_{}".format(input_a)))
            for input_b in range(w):
                localB.append(hcl.compute((1,), lambda x : B[k, input_b + w * b_col], "localB_{}".format(input_b)))

            # systolic connection
            net = [[None] * h] * w
            for i in range(h + w - 1):
                for row in range(i + 1):
                    col = i - row
                    if col < 0 or col > w-1 or row > h-1: continue
                    ## instantiate a PE and record partial results
                    input_a = localA[row] if col == 0 else hcl.compute((1,), lambda x : net[row][col-1][0], "input_a{}{}".format(row, col))
                    input_b = localB[col] if row == 0 else hcl.compute((1,), lambda x : net[row-1][col][1], "input_b{}{}".format(row, col))

                    out = hcl.compute((3,), lambda x : 0, "out_{}{}".format(row, col))
                    pe[row * w + col](input_a, input_b, out)
                    hcl.print((out[2], input_a, input_b), "out=%d a=%d b=%d\n")

                    # MAC adds up the intermediate result
                    O[row + h * b_row, col + w * b_col] += out[2]
                    net[row][col] = out

        block_rows = int(m / h)
        block_cols = int(n / w)
        O = hcl.compute((m, n), lambda *args : 0, name="Output")
        hcl.mutate((block_rows, block_cols, k), lambda b_row, b_col, k: update(b_row, b_col, k, O), name="update")
        return O

    s = hcl.create_schedule([Ain, Bin], systolic_array)

    # data_graph = s.dataflow_graph(plot=True)
    # pipeline loop
    k = systolic_array.update
    s[k].pipeline(k.axis[0])
    s[k].pipeline(k.axis[1])
    s[k].pipeline(k.axis[2])

    # systolic connection with .to()
    # s.to(k.input_b11, s[k.out_11])

    if isinstance(target, hcl.platform):
        s.to([Ain, Bin], target.xcel)
        s.to(systolic_array.update.Output, target.host)
        target.config(compile="vivado_hls", mode="csyn")

    print(hcl.lower(s))
    return hcl.build(s, target=target)


# matrix size
# m*k k*n
m = 4
k = 4
n = 4
# systolic size
w = 2
h = 2

np_1 = np.random.randint(10, size=(m,k))
np_2 = np.random.randint(10, size=(k,n))

dtype = hcl.Int()

hcl_m1 = hcl.asarray(np_1, dtype=dtype)
hcl_m2 = hcl.asarray(np_2, dtype=dtype)
hcl_m3 = hcl.asarray(np.zeros((m,n)), dtype=dtype)

# systolic array
# target = hcl.platform.zc706
# target = "vhls_csim"
target = "llvm"
# target = "vhls"
fs = systolic(m, k, n, w, h, dtype=hcl.Int(), target=target)
if isinstance(target, str) and target == "vhls":
    print(fs)
else:
    fs(hcl_m1, hcl_m2, hcl_m3)
print("Systolic Array's result = ")
print(hcl_m3.asnumpy())

answer = np.dot(np_1, np_2)
print("Correct Answer = ")
print(answer)


if np.array_equal(hcl_m3.asnumpy(), answer):
    print("Yeah we got that right")
else:
    print("And I Oop...")


