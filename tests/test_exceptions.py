# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import warnings


def test_deprecate_stage():
    with warnings.catch_warnings(record=True) as w:
        hcl.init(hcl.Int(32))
        A = hcl.placeholder((10,))

        def kernel(A):
            with hcl.Stage("B"):
                with hcl.for_(0, 10) as i:
                    A[i] = A[i] + 1
            return

        s = hcl.create_schedule([A], kernel)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message)
