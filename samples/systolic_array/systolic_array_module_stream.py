import heterocl as hcl
import numpy as np

def systolic(m=2, k=2, n=2, w=2, h=2, dtype=hcl.Int(), target=None):

    hcl.init(dtype)

    Ain = hcl.placeholder((m, k), dtype=dtype, name="A")
    Bin = hcl.placeholder((k, n), dtype=dtype, name="B")
    # Output = hcl.placeholder((m, n), dtype=dtype, name="output")

    def systolic_array(A, B):

        # define modules with new decorator
        @hcl.def2_([(1,), (1,), (1,), (1,), (1,)], amount=w*h)
        def pe(a, b, out_a, out_b, out):
            out_a.v = a
            out_b.v = b
            out.v = a * b

        # each k calls of update function calculate one block of result matrix
        # b_row: block row index
        # b_col: block col index

        block_rows = int(m / h)
        block_cols = int(n / w)
        O = hcl.compute((m, n), lambda *args : 0, name="Output")

        with hcl.Stage("update") as s:
            with hcl.for_(0, block_rows, name="row") as b_row:
                with hcl.for_(0, block_cols, name="col") as b_col:

                    # stage 1: create pipes
                    localA = []
                    localB = []
                    p_out = hcl.compute((h,w), lambda y, x: 0, "localO")
                    for input_a in range(h):
                        localA.append(hcl.compute((k,), lambda x: 0, "localA_{}".format(input_a)))
                    for input_b in range(w):
                        localB.append(hcl.compute((k,), lambda x: 0, "localB_{}".format(input_b)))

                    # systolic connection
                    net_a = [[0 for x in range(h)] for y in range(w)]
                    net_b = [[0 for x in range(h)] for y in range(w)]
                    for i in range(h + w - 1):
                        for row in range(i + 1):
                            col = i - row
                            if col < 0 or col > w-1 or row > h-1: continue
                            out_a = hcl.scalar(0, "out_a_{}{}".format(row, col))
                            out_b = hcl.scalar(0, "out_b_{}{}".format(row, col))
                            net_a[row][col] = out_a
                            net_b[row][col] = out_b

                    with hcl.for_(0, k, name="_k") as _k:
                        for input_a in range(h):
                            localA[input_a][_k] = A[input_a + h * b_row, _k] 
                        for input_b in range(w):
                            localB[input_b][_k] = B[_k, input_b + w * b_col] 

                    # stage 2: do computation
                    with hcl.for_(0, k, name="k") as k_:
                        # save the partial sum into local buffer 
                        for i in range(h + w - 1):
                            for row in range(i + 1):
                                col = i - row
                                if col < 0 or col > w-1 or row > h-1: continue
                                ## instantiate a PE and record partial results
                                input_a, input_b = 0, 0
                                if col == 0:
                                    temp_a = hcl.scalar(localA[row][k_], "in_a{}0".format(row))
                                    input_a = temp_a
                                else:
                                    input_a = net_a[row][col-1]
                                if row == 0:
                                    temp_b = hcl.scalar(localB[col][k_], "in_b0{}".format(col))
                                    input_b = temp_b
                                else:
                                    input_b = net_b[row-1][col]

                                out   = hcl.scalar(0, "out_{}{}".format(row, col))
                                out_a = net_a[row][col]
                                out_b = net_b[row][col]

                                pe[row * w + col](input_a, input_b, net_a[row][col], out_b, out)
                                # hcl.print((out[2], input_a, input_b), "out=%d a=%d b=%d\n")

                                # MAC adds up the intermediate result
                                p_out[row][col] += out

                    # stage 3: push results to output
                    with hcl.for_(0, h, name="h_") as h_:
                        with hcl.for_(0, w, name="w_") as w_:
                            O[h_ + h * b_row, w_ + w * b_col] = p_out[h_, w_]
            
        return O

    s = hcl.create_schedule([Ain, Bin], systolic_array)
    # pipeline loop
    k = systolic_array.update
    s[k].pipeline(k.axis[2])

    # inter-stage streaming 
    for input_a in range(h):
        t = getattr(k, "localA_{}".format(input_a))
        s.to(t, s[k], s[k], depth=1)
    for input_b in range(w):
        t = getattr(k, "localB_{}".format(input_b))
        s.to(t, s[k], s[k], depth=1)

    # inter-pe streaming channels
    for i in range(h + w - 1):
        for row in range(i + 1):
            col = i - row
            if col < 0 or col > w-2 or row > h-2 : continue
            ta = getattr(k, "out_a_{}{}".format(row, col))
            tb = getattr(k, "out_b_{}{}".format(row, col))
            s.to(ta, s[k], s[k])
            s.to(tb, s[k], s[k])

    if isinstance(target, hcl.platform):
        s.to([Ain, Bin], target.xcel)
        s.to(systolic_array.update.Output, target.host)
        target.config(compile="vitis", mode="hw_exe")

    # print(hcl.lower(s))
    return hcl.build(s, target=target)

# matrix size
# m*k k*n
m = 32
k = 32
n = 32
# systolic size
w = 4
h = 4

np_1 = np.random.randint(10, size=(m,k))
np_2 = np.random.randint(10, size=(k,n))

dtype = hcl.Int()

hcl_m1 = hcl.asarray(np_1, dtype=dtype)
hcl_m2 = hcl.asarray(np_2, dtype=dtype)
hcl_m3 = hcl.asarray(np.zeros((m,n)), dtype=dtype)

# systolic array
target = hcl.platform.zc706
# target = "vhls_csim"
# target = "llvm"
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


