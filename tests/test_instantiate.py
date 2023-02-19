# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import pytest


# Instance A: vadd
def vadd(A, B):
    C = hcl.compute((10,), lambda i: A[i] + B[i], name="C")
    s = hcl.customize([A, B])
    # add customization
    s[C].unroll(C.axis[0])
    return C


# Instance B: vmul
def vmul(A, B):
    C = hcl.compute((10,), lambda i: A[i] * B[i], name="C")
    s = hcl.customize([A, B])
    # add customization
    s[C].unroll(C.axis[0])
    return C


def kernel(A, B, C, D):
    vadd_inst = hcl.instantiate(vadd, "vadd_inst")
    vmul_inst = hcl.instantiate(vmul, "vmul_inst")
    E = vadd_inst(A, B)
    F = vmul_inst(C, D)
    return E, F


@pytest.mark.skip("instantiation not supported")
def test_instantiate():
    A = hcl.placeholder((10,), name="A")
    B = hcl.placeholder((10,), name="B")
    C = hcl.placeholder((10,), name="C")
    D = hcl.placeholder((10,), name="D")
    s = hcl.customize([A, B, C, D], kernel)
    assert len(s.instance_modules) == 2
