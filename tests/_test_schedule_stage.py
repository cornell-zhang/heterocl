import heterocl as hcl
import numpy as np

def test_compute_single_stage():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x]+1, "B")
    s = hcl.create_schedule(A, kernel)

    assert kernel.B.input_stages == set([A.first_update])

def test_update_single_stage():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        hcl.update(A, lambda x: A[x]+1, "AU")
    s = hcl.create_schedule(A, kernel)

    assert kernel.AU.input_stages == set([A.first_update])

def test_compute_two_stages():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x: A[x]+1, "B")
        return hcl.compute(B.shape, lambda x: B[x]+1, "C")
    s = hcl.create_schedule(A, kernel)

    assert kernel.C.input_stages == set([kernel.B])

def test_compute_two_stages_complex():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x: A[x]+1, "B")
        return hcl.compute(B.shape, lambda x: A[x]+B[x], "C")
    s = hcl.create_schedule(A, kernel)

    assert kernel.C.input_stages == set([kernel.B, A.first_update])

def test_imperative_stage_rhs():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0] += 1
    s = hcl.create_schedule(A, kernel)

    assert kernel.S.input_stages == set([A.first_update])

def test_imperative_stage_lhs():

    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    def kernel(A, B):
        with hcl.Stage("S"):
            A[0] = B[0]
    s = hcl.create_schedule([A, B], kernel)

    assert kernel.S.input_stages == set([A.first_update, B.first_update])

def test_imperative_multi_stages():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x: A[x]+1, "B")
        C = hcl.compute(A.shape, lambda x: A[x]+1, "C")
        with hcl.Stage("S"):
            C[0] = B[0]
        return B, C
    s = hcl.create_schedule(A, kernel)

    assert kernel.S.input_stages == set([kernel.B, kernel.C])

