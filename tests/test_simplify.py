import heterocl as hcl
import pytest
from hcl_mlir.exceptions import APIError


def test_remove_single_loop():
    hcl.init()
    a = hcl.placeholder((1,))
    b = hcl.compute(a.shape, lambda x: a[x] + 1)
    s = hcl.create_schedule([a])
    ir = hcl.lower(s)
    assert "0 to 1" not in str(ir)


def test_simplify_slice():
    with pytest.raises(APIError):
        hcl.init()
        A = hcl.placeholder((10,), "A")

        def kernel(A):
            A[5][2:2] = 4

        s = hcl.create_schedule(A, kernel)
        ir = hcl.lower(s)
