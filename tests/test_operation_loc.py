# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import pytest

def test_compute():
    shape = (1,10)
    def loop_body(data, power):
        E = hcl.compute(shape, lambda x,y: hcl.power(data[x,y],power[x,y]), "loop_body")
        print(E.loc)
        assert np.equal(str(E.loc),"operation.py:472")

        return E

    A = hcl.placeholder(shape, "A")
    B = hcl.placeholder(shape, "B")
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

    print(A.loc)
    print(B.loc)

    assert np.equal(str(A.loc),"test_ast_loc.py:21") and np.equal(str(B.loc),"test_ast_loc.py:22")