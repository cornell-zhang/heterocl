import tvm

A = tvm.placeholder((10, 10), name = "A")
B = tvm.placeholder((10, 10), name = "B")

C = tvm.compute(A.shape, lambda x, y: A[x, y] + B[x, y], name = "C")

s = tvm.create_schedule(C.op)

func = tvm.build(s, [A, B, C], name = 'add', target_host = 'c')

print func.get_source()
