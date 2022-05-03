import heterocl as hcl

def reduce(b, w, q):
    bw  = hcl.get_bitwidth(q.dtype)
    bwh = bw // 2
    mask = (1 << bwh) - 1

    b = hcl.scalar(b, "b", dtype=hcl.UInt(bw))
    w = hcl.scalar(w, "w", dtype=hcl.UInt(bw))
    q = hcl.scalar(q, "q", dtype=hcl.UInt(bw))

    a = w * b
    for i in range(2):
        t = (-a) & mask
        s = (a + (t*q)) >> bwh
        a = s
    a = hcl.select(a < q, a, a - q)
    res = hcl.scalar(a, "reduce", dtype=hcl.UInt(bw))
    return res


def test():

    def func(A,B):
        B[0] = reduce(A[0], A[1], A[2])

    A = hcl.placeholder((5,), "A", dtype=hcl.UInt(32))
    B = hcl.placeholder((2,), "B", dtype=hcl.UInt(32))
    s = hcl.create_schedule([A,B], func)

    code = hcl.build(s, "shls")
    assert "(sc_biguint<131>)" in code

if __name__ == "__main__":
    test()