# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import re
import pytest


def test_compute_single_stage():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1, "B")

    s = hcl.create_schedule(A, kernel)

    node_map = s.DataflowGraph.node_map
    assert node_map["A"] in node_map["B"].parents


def test_update_single_stage():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        hcl.update(A, lambda x: A[x] + 1, "AU")

    s = hcl.create_schedule(A, kernel)

    node_map = s.DataflowGraph.node_map
    assert node_map["A"] in node_map["AU"].parents


def test_compute_two_stages():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda x: A[x] + 1, "B")
        return hcl.compute(B.shape, lambda x: B[x] + 1, "C")

    s = hcl.create_schedule(A, kernel)

    node_map = s.DataflowGraph.node_map
    assert node_map["B"] in node_map["C"].parents


def test_compute_two_stages_complex():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda x: A[x] + 1, "B")
        return hcl.compute(B.shape, lambda x: A[x] + B[x], "C")

    s = hcl.create_schedule(A, kernel)

    node_map = s.DataflowGraph.node_map
    assert (
        node_map["A"] in node_map["C"].parents
        and node_map["B"] in node_map["C"].parents
    )


def test_imperative_stage_rhs():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        A[0] += 1

    s = hcl.create_schedule(A, kernel)
    hcl.lower(s)
    print(s.module)
    assert r"%arg0[0]" in str(s.module)


def test_imperative_stage_lhs():
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")

    def kernel(A, B):
        A[0] = B[0]

    s = hcl.create_schedule([A, B], kernel)
    hcl.lower(s)
    assert r"%arg1[0]" in str(s.module)


def test_imperative_multi_stages():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda x: A[x] + 1, "B")
        C = hcl.compute(A.shape, lambda x: A[x] + 1, "C")
        C[0] = B[0]
        return B, C

    s = hcl.create_schedule(A, kernel)
    hcl.lower(s)
    node_map = s.DataflowGraph.node_map
    assert (
        node_map["A"] in node_map["B"].parents
        and node_map["A"] in node_map["C"].parents
    )
    assert re.search(r'affine\.load %\d+\[0\] {from = "B"}', str(s.module))
