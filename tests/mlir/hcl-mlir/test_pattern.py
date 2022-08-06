from pyrsistent import b
import heterocl as hcl
from heterocl.pattern import *


@is_rewrite
def expr_rewrite(a, b, c, res):
    new_res = c + a * b
    replace(res, new_res)


@is_pattern(benefit=0)
def pattern_rewrite():
    dtype = hcl.Int(32)
    a = value(dtype)
    b = value(dtype)
    c = value(dtype)
    res = a * b + c
    expr_rewrite(a, b, c, res)


@is_transform
def loop_transform(res):
    loop = parent_loop(res, 1)
    outer_loop, inner_loop = split(loop, 2)
    unroll(inner_loop, 2)
    pipeline(outer_loop, 1)


@is_pattern(benefit=0)
def pattern_transform1():
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
def pattern_transform2(loop: hcl.OpHandle):
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

    s.apply(pattern_rewrite)
    s.apply(pattern_transform1)
    s.apply(pattern_transform2, s[gemm.C])
    print(s.device_module)

    f = hcl.build(s, "vhls")
    print(f)


if __name__ == '__main__':
    main()
