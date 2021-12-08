import heterocl as hcl
import numpy as np
hcl.init()

def test_int7(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x4A", "v", dtype=hcl.Int(7))
        return v.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.Int(7))

def test_uint7(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x4A", "v", dtype=hcl.UInt(7))
        return v.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.UInt(7))

def test_int15(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x40FF", "v", dtype=hcl.Int(15))
        return v.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.Int(15))

def test_uint15(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x40FF", "v", dtype=hcl.UInt(15))
        return v.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.UInt(15))

def test_int31(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x4F0000FF", "v", dtype=hcl.Int(31))
        return v.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.Int(31))

def test_uint31(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x4F0000FF", "v", dtype=hcl.UInt(31))
        return v.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.UInt(31))

def test_int62(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x2A000000FF0000FF", "v", dtype=hcl.Int(62))
        return v.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.Int(62))

def test_uint62(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x2A000000FF0000FF", "v", dtype=hcl.UInt(62))
        return v.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.UInt(62))

A_int7 = hcl.placeholder((1,), "A_int7", dtype=hcl.Int(7))
s_int7 = hcl.create_schedule([A_int7], test_int7)
m_int7 = hcl.build (s_int7)

A_uint7 = hcl.placeholder((1,), "A_uint7", dtype=hcl.UInt(7))
s_uint7 = hcl.create_schedule([A_uint7], test_uint7)
m_uint7 = hcl.build (s_uint7)

A_int15 = hcl.placeholder((1,), "A_int15", dtype=hcl.Int(15))
s_int15 = hcl.create_schedule([A_int15], test_int15)
m_int15 = hcl.build (s_int15)

A_uint15 = hcl.placeholder((1,), "A_uint15", dtype=hcl.UInt(15))
s_uint15 = hcl.create_schedule([A_uint15], test_uint15)
m_uint15 = hcl.build (s_uint15)

A_int31 = hcl.placeholder((1,), "A_int31", dtype=hcl.Int(31))
s_int31 = hcl.create_schedule([A_int31], test_int31)
m_int31 = hcl.build (s_int31)

A_uint31 = hcl.placeholder((1,), "A_uint31", dtype=hcl.UInt(31))
s_uint31 = hcl.create_schedule([A_uint31], test_uint31)
m_uint31 = hcl.build (s_uint31)

A_int62 = hcl.placeholder((1,), "A_int62", dtype=hcl.Int(62))
s_int62 = hcl.create_schedule([A_int62], test_int62)
m_int62 = hcl.build (s_int62)

A_uint62 = hcl.placeholder((1,), "A_uint62", dtype=hcl.UInt(62))
s_uint62 = hcl.create_schedule([A_uint62], test_uint62)
m_uint62 = hcl.build (s_uint62)

A_int7 = hcl.asarray([0xA0A0], dtype=A_int7.dtype)
R_int7 = hcl.asarray([99], dtype=hcl.Int(7))
m_int7(A_int7, R_int7)

A_uint7 = hcl.asarray([0xA0A0], dtype=A_uint7.dtype)
R_uint7 = hcl.asarray([99], dtype=hcl.UInt(7))
m_uint7(A_uint7, R_uint7)

A_int15 = hcl.asarray([0xA0A0], dtype=A_int15.dtype)
R_int15 = hcl.asarray([99], dtype=hcl.Int(15))
m_int15(A_int15, R_int15)

A_uint15 = hcl.asarray([0xA0A0], dtype=A_uint15.dtype)
R_uint15 = hcl.asarray([99], dtype=hcl.UInt(15))
m_uint15(A_uint15, R_uint15)

A_int31 = hcl.asarray([0xA0A0], dtype=A_int31.dtype)
R_int31 = hcl.asarray([99], dtype=hcl.Int(31))
m_int31(A_int31, R_int31)

A_uint31 = hcl.asarray([0xA0A0], dtype=A_uint31.dtype)
R_uint31 = hcl.asarray([99], dtype=hcl.UInt(31))
m_uint31(A_uint31, R_uint31)

A_int62 = hcl.asarray([0xA0A0], dtype=A_int62.dtype)
R_int62 = hcl.asarray([99], dtype=hcl.Int(62))
m_int62(A_int62, R_int62)

A_uint62 = hcl.asarray([0xA0A0], dtype=A_uint62.dtype)
R_uint62 = hcl.asarray([99], dtype=hcl.UInt(62))
m_uint62(A_uint62, R_uint62)

print(f"R_int7 = {[bin(i) for i in R_int7.asnumpy()]}")
print(f"R_uint7 = {[hex(i) for i in R_uint7.asnumpy()]}")
print(f"R_int15 = {[hex(i) for i in R_int15.asnumpy()]}")
print(f"R_uint15 = {[hex(i) for i in R_uint15.asnumpy()]}")
print(f"R_int31 = {[hex(i) for i in R_int31.asnumpy()]}")
print(f"R_uint31 = {[hex(i) for i in R_uint31.asnumpy()]}")
print(f"R_int62 = {[hex(i) for i in R_int62.asnumpy()]}")
print(f"R_uint62 = {[hex(i) for i in R_uint62.asnumpy()]}")

def test_int127_lower(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x8ABCFFFFFFBAFFFA000A", "v", dtype=hcl.UInt(127))
        b = hcl.scalar(v >> 64, "b", dtype=hcl.UInt(63))
        c = hcl.scalar(v & 0xFFFFFFFFFFFFFFFF, "c", dtype=hcl.UInt(64))
        return c.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.UInt(64))

def test_int127_upper(A):
    def doit(x):
        x = 0xFA_FF00_FFFF
        v = hcl.scalar("0x8ABCFFFFFFBAFFFA000A", "v", dtype=hcl.UInt(127))
        b = hcl.scalar(v >> 64, "b", dtype=hcl.UInt(63))
        c = hcl.scalar(v & 0x7FFFFFFFFFFFFFFF, "c", dtype=hcl.UInt(64))
        return b.v
    return hcl.compute(A.shape, lambda i: doit(i), "doit", dtype=hcl.UInt(63))

A = hcl.placeholder((1,), "A", dtype=hcl.UInt(63))
s_lower = hcl.create_schedule([A], test_int127_lower)
s_upper = hcl.create_schedule([A], test_int127_upper)
m_lower = hcl.build(s_lower)
m_upper = hcl.build(s_upper)

hcl_A = hcl.asarray([0], hcl.UInt(63))
hcl_R_lower = hcl.asarray([0], hcl.UInt(64))
hcl_R_upper = hcl.asarray([0], hcl.UInt(63))

m_lower(hcl_A, hcl_R_lower)
m_upper(hcl_A, hcl_R_upper)

print(f"hcl_R_lower = {[hex(i) for i in hcl_R_lower.asnumpy()]}")
print(f"hcl_R_upper = {[hex(i) for i in hcl_R_upper.asnumpy()]}")

