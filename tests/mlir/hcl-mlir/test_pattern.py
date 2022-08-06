import heterocl as hcl


def pattern1(p: hcl.Pattern):
    dtype = hcl.Int(32)

    a = p.value(dtype)
    b = p.value(dtype)
    c = p.value(dtype)
    res = a * b + c

    res = p.start_transform(res)
    target_loop = p.get_parent_loop(res, 1)
    outer_loop, inner_loop = p.split(target_loop, 2)
    p.unroll(inner_loop, 2)
    p.pipeline(outer_loop, 1)
    p.end_transform_or_rewrite()


def pattern2(p: hcl.Pattern, loop: hcl.OpHandle):
    target_loop = p.start_transform(loop)
    p.unroll(target_loop, 2)
    p.end_transform_or_rewrite()


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

    p = s.apply("pattern1", 0, pattern1)
    # p = s.apply("pattern2", 0, pattern2, s[gemm.C])
    print(s.device_module)

    # f = hcl.build(s, "vhls")
    # print(f)


if __name__ == '__main__':
    main()
