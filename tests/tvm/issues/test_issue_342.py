import heterocl as hcl
import numpy as np

def test_numpy_operator_err_msg_0():
    hcl.init()

    A = hcl.placeholder((10, 10))
    np_A = np.array([5, 10, 15])

    try:
        np_A[1] > A[5, 5]
    except hcl.debug.APIError:
        return
    assert False

def test_numpy_operator_err_msg_1():
    hcl.init()

    A = hcl.placeholder((10, 10))
    np_A = np.array([5, 10, 15])

    try:
        np_A[1] + A[5, 5]
    except hcl.debug.APIError:
        return
    assert False

def test_numpy_operator_no_err_0():
    hcl.init()

    A = hcl.placeholder((10, 10))
    np_A = np.array([5, 10, 15])

    A[5, 5] > np_A[1]

def test_numpy_operator_no_err_1():
    hcl.init()

    A = hcl.placeholder((10, 10))
    np_A = np.array([5, 10, 15])

    A[5, 5] + np_A[1]
