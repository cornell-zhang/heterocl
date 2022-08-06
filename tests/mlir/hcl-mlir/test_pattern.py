import heterocl as hcl
from heterocl.pattern import *


@is_transform
def loop_transform(target):
    target_loop = parent_loop(target, 1)
    outer_loop, inner_loop = split(target_loop, 2)
    unroll(inner_loop, 2)
    pipeline(outer_loop, 1)


@is_pattern(benefit=0)
def pattern1():
    dtype = hcl.Int(32)
    a = value(dtype)
    b = value(dtype)
    c = value(dtype)
    res = a * b + c
    loop_transform(res)


@is_transform
def loop_unroll(loop):
    unroll(loop, 2)


@is_pattern(benefit=0)
def pattern2(loop: hcl.OpHandle):
    loop_unroll(loop)


def main(M=32, N=32, K=32):
    hcl.init(hcl.Int())
    A = hcl.placeholder((M, K), name="A")
    B = hcl.placeholder((K, N), name="B")

    def gemm(A, B):
        k = hcl.reduce_axis(0, K, name="k")
        C = hcl.compute((32, 32), lambda i, j:
                        hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    s = hcl.create_schedule([A, B], gemm)

    p = s.apply(pattern1)
    # p = s.apply(pattern2, s[gemm.C])
    print(s.device_module)

    f = hcl.build(s, "vhls")
    print(f)


if __name__ == '__main__':
    main()
