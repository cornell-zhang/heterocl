import heterocl as hcl
from montgomery import *
from random import sample


def old_pattern():
    dtype = hcl.Int(32)
    attr_type = hcl.Int(64)
    bool_type = hcl.Int(1)

    # Require v == -1
    v = hcl.pdl_value(dtype)
    cm1 = hcl.pdl_op('arith.constant', [], {
                     'value': hcl.pdl_attr(-1, dtype)}, [hcl.pdl_type(dtype)])
    cond0 = hcl.pdl_op('arith.cmpi', [v, cm1], {
                       'predicate': hcl.pdl_attr(0, attr_type)},
                       [hcl.pdl_type(bool_type)])
    req0 = hcl.pdl_op('fdv.require', [cond0])

    # Require z + 1 == 1 << k, name S = z + 1
    z = hcl.pdl_value(dtype)
    c1 = hcl.pdl_op('arith.constant', [], {
                    'value': hcl.pdl_attr(1, dtype)}, [hcl.pdl_type(dtype)])
    S = hcl.pdl_op('arith.addi', [z, c1], type_handles=[hcl.pdl_type(dtype)])

    k = hcl.pdl_value(dtype)
    shl = hcl.pdl_op('arith.shli', [c1, k], type_handles=[hcl.pdl_type(dtype)])
    cond1 = hcl.pdl_op('arith.cmpi', [S, shl], {
                       'predicate': hcl.pdl_attr(0, attr_type)},
                       [hcl.pdl_type(bool_type)])
    req1 = hcl.pdl_op('fdv.require', [cond1])

    # Match a * v + z
    a = hcl.pdl_value(dtype)
    mul = hcl.pdl_op('arith.muli', [a, v], type_handles=[hcl.pdl_type(dtype)])
    and0 = hcl.pdl_op('arith.addi', [mul, z],
                      type_handles=[hcl.pdl_type(dtype)])

    # Rewrite into S - a & z
    hcl.pdl_rewrite(and0)
    and1 = hcl.pdl_op('arith.andi', [a, z], type_handles=[hcl.pdl_type(dtype)])
    sub = hcl.pdl_op('arith.subi', [S, and1],
                     type_handles=[hcl.pdl_type(dtype)])
    hcl.pdl_replace(and0, sub)


def pattern1(p: hcl.Pattern):
    dtype = hcl.Int(32)

    v = p.value(dtype)
    # p.require(lambda v: v == -1, v)

    z = p.value(dtype)
    # S = z + 1
    # k = p.value(dtype)
    # p.require(lambda S, k: S == 1 << k, S, k)

    a = p.value(dtype)
    res = z + a * v

    # p.start_rewrite(res)
    # new_res = S - (a & z)
    # p.replace(res, new_res)

    res = p.start_transform(res)
    target_loop = p.get_parent_loop(res)
    p.loop_unroll(target_loop, 2)
    p.end_transform_or_rewrite()


def pattern2(p: hcl.Pattern, loop: hcl.OpHandle):
    target_loop = p.start_transform(loop)
    p.loop_unroll(target_loop, 2)
    p.end_transform_or_rewrite()


def main():
    hcl.init("uint1")
    shape = (10,)
    BW = 32
    minM, maxM = (1 << BW - 1) + 1, (1 << BW) - 1
    A, B, C = (
        vtype('A', maxM, shape=shape),
        vtype('B', maxM, shape=shape),
        vtype('C', 0, shape=shape, bits=BW)
    )
    tensors = [A, B, C] + list(montgomery_fun.twiddles(maxM))
    placeholders = [t.placeholder() for t in tensors]
    Ap, Mp, vp, kp, zp = vtype.unpack(placeholders, "A M v k z")

    def montgomery_inv(A, M, v, k, z):
        def inv(i):
            s = hcl.cast(z.dtype, (A[i] * v.v) & z.v)
            r = (A[i] + s * M.v) >> k.v
            return hcl.select(r < M.v, r, r - M.v)
        C = hcl.compute(A.shape, inv, "C", M.dtype)
        return C

    s = hcl.create_schedule([Ap, Mp, vp, kp, zp], montgomery_inv)
    print(s.device_module)

    # p = s.apply("pattern1", 0, pattern1)
    p = s.apply("pattern2", 0, pattern2, s[C])
    print(s.device_module)


if __name__ == '__main__':
    main()
