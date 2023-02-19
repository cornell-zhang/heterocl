# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from re import I
import heterocl as hcl
import numpy as np
from itertools import permutations
import os
import pytest
import hcl_mlir


# Test DFG partitioning
def test_move_outputs():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] * 2, "B")
        hcl.update(B, lambda i, j: B[i, j] + 1, "update1")
        hcl.update(B, lambda i, j: B[i, j] * 2, "update2")
        return B

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls", mode="csim", project="stream_tests/test_move_outputs.prj"
    )
    s = hcl.create_schedule([A], kernel)
    s.to(A, target.xcel)
    s.to(kernel.update1.B, target.host)
    mod = hcl.build(s, target)
    assert "top" in mod.host_src
    assert "B" in mod.src
    assert "update1" in mod.src
    assert "update2" in mod.host_src


def test_in_place_update():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")

    def kernel(A):
        hcl.update(A, lambda i, j: A[i, j] + 1, "update1")
        hcl.update(A, lambda i, j: A[i, j] * 2, "update2")

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls",
        mode="csim",
        project="stream_tests/test_in_place_update.prj",
    )
    s = hcl.create_schedule([A], kernel)
    s.to(A, target.xcel)
    s.to(kernel.update1.A, target.host)

    mod = hcl.build(s, target)
    assert "top" in mod.host_src  # check top function call is there in host code
    assert "update1" in mod.src  # check update1 is there in device code
    assert "update2" in mod.host_src  # check update2 function is there in host code


def test_multiple_subgraph():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "C")
        D = hcl.compute(C.shape, lambda i, j: B[i, j] + 1, "D")
        return hcl.compute(C.shape, lambda i, j: C[i, j] + D[i, j], "E")

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls",
        mode="csim",
        project="stream_tests/test_multiple_subgraph.prj",
    )
    s = hcl.create_schedule([A, B], kernel)
    s.to([A, B], target.xcel)
    s.to([kernel.E], target.host)
    mod = hcl.build(s, target)
    assert "top" in mod.host_src
    assert "C" in mod.src
    assert "D" in mod.src
    assert "E" in mod.src


def test_extern_ops():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda *args: A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args: B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args: C[args] * 2, "D")
        return D

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls", mode="csim", project="stream_tests/test_extern_ops.prj"
    )
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    mod = hcl.build(s, target)
    assert "top" in mod.host_src
    assert "B" in mod.host_src
    assert "C" in mod.src
    assert "D" in mod.host_src


@pytest.mark.skip(reason="TypeError: .to() got an unexpected keyword argument axis")
def test_inner_loop_body_placement():
    def _test_imperative_loop():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda *args: 0, "C")
            with hcl.for_(0, 10, tag="stage") as i:
                with hcl.for_(0, 32) as j:
                    B[i, j] = A[i, j] + B[i, j]
                    C[i, j] = 2 * B[i, j]
            return C

        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A, B], kernel)

        stage = kernel.stage
        s.to(stage, target.xcel, axis=1)
        code = str(hcl.lower(s))
        pattern = "test({}, {}, {}, {})"
        combination = [
            pattern.format(*_) for _ in list(permutations(["A", "B", "C", "i"]))
        ]
        cond = any([_ in code for _ in combination])
        assert cond, code

    def _test_declarative_loop():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")

        def kernel(A):
            C = hcl.compute(A.shape, lambda *args: A[args] * 4, "C")
            return C

        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.C, target.xcel, axis=1)
        code = str(hcl.lower(s))
        assert "test(C, A, args)" in code

    def _test_inner_loop_tile():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")

        def kernel(A):
            C = hcl.compute(A.shape, lambda *args: A[args] * 4, "C")
            return C

        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)

        stage = kernel.C
        yo, yi = s[stage].split(stage.axis[0], factor=3)
        xo, xi = s[stage].split(stage.axis[1], factor=3)
        s.to(kernel.C, target.xcel, axis=1)
        code = str(hcl.lower(s))
        assert "test(args.outer, C, A)" in code

    _test_imperative_loop()
    _test_declarative_loop()
    _test_inner_loop_tile()


# Test on-chip data movement
@pytest.mark.skip(reason="IndexError: list index out of range")
def test_stages_one_to_many():
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, lambda i, j: C[i][j] + 1, "D")
        E = hcl.compute(C.shape, lambda i, j: C[i][j] * 2, "E")
        return D, E

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls",
        mode="csim",
        project="stream_tests/test_stages_one_to_many.prj",
    )
    s = hcl.create_schedule([A, B], kernel)
    s.to(kernel.C, s[kernel.D])
    s.to(kernel.C, s[kernel.E])

    mod = hcl.build(s, target)
    assert "Stage_C" in mod.host_src
    assert "Stage_D" in mod.host_src
    assert "Stage_E" in mod.host_src


def test_mixed_stream():
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, lambda i, j: C[i][j], "D")
        return D

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls", mode="csim", project="stream_tests/test_mixed_stream.prj"
    )
    s = hcl.create_schedule([A, B], kernel)

    s.to([A, B], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.C, s[kernel.D])

    mod = hcl.build(s, target)
    assert "top" in mod.host_src
    assert "Stage_C" in mod.src
    assert "Stage_D" in mod.src


@pytest.mark.skip(reason="IndexError: list index out of range")
def test_fork_join():
    def inter_stage_fork():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: A[i, j] + B[i, j], "C")
            D = hcl.compute(C.shape, lambda i, j: C[i, j] + 1, "D")
            E = hcl.compute(C.shape, lambda i, j: C[i, j] * 2, "E")
            return D, E

        target = hcl.Platform.aws_f1
        target.config(
            compiler="vivado_hls",
            mode="csim",
            project="stream_tests/test_inter_stage_fork.prj",
        )
        s = hcl.create_schedule([A, B], kernel)
        s.to(kernel.C, [kernel.D, kernel.E])
        mod = hcl.build(s, target)
        assert "top" in mod.host_src
        assert "Stage_C" in mod.host_src
        assert "Stage_D" in mod.host_src

    # Create channels but enforce the dependency
    def inter_stage_join():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: 0, "C")
            hcl.update(C, lambda i, j: A[i, j] + 1, "s1")
            hcl.update(C, lambda i, j: B[i, j] * 2, "s2")
            return hcl.compute(C.shape, lambda *args: C[args] + 3, "ret")

        target = hcl.Platform.aws_f1
        target.config(
            compiler="vivado_hls",
            mode="csim",
            project="stream_tests/test_inter_stage_join.prj",
        )
        s = hcl.create_schedule([A, B], kernel)

        s.to(kernel.s1.C, kernel.ret.C)
        s.to(kernel.s2.C, kernel.ret.C)

        mod = hcl.build(s, target)
        assert "top" in mod.host_src

    inter_stage_fork()
    inter_stage_join()


@pytest.mark.skip(reason="Schedule object has no attribute subgraph")
def test_kernel_duplicate():
    def test_extract_subgraph(combine=False):
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: 0, "C")
            hcl.update(C, lambda i, j: A[i, j] + 1, "s1")
            hcl.update(C, lambda i, j: B[i, j] * 2, "s2")
            return hcl.compute(C.shape, lambda *args: C[args] + 3, "ret")

        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A, B], kernel)

        s.to([A, B], target.xcel)
        s.to(kernel.ret, target.host)

        # Combine and split
        if combine == True:
            # Merge stages from top to bottom
            s[kernel.C].compute_at(s[kernel.s1], kernel.s1.axis[1])
            s[kernel.s1].compute_at(s[kernel.s2], kernel.s2.axis[1])
            s[kernel.s2].compute_at(s[kernel.ret], kernel.ret.axis[1])
            s[kernel.ret].split(kernel.ret.axis[0], factor=2)

        code = str(hcl.lower(s))
        assert 'io attr: "A"' in code
        assert 'io attr: "B"' in code
        assert 'io attr: "ret"' in code

    def test_merge_kernel_stages():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: 0, "C")
            hcl.update(C, lambda i, j: A[i, j] + 1, "s1")
            hcl.update(C, lambda i, j: B[i, j] * 2, "s2")
            return hcl.compute(C.shape, lambda *args: C[args] + 3, "ret")

        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A, B], kernel)

        s.to([A, B], target.xcel)
        s.to(kernel.ret, target.host)
        tops = s.subgraph()[0]
        dev_body = str(tops.op.body)
        assert 'device_scope = "fpga"' in dev_body

    test_merge_kernel_stages()
    test_extract_subgraph(True)


# Test cross device data movement
@pytest.mark.skip(reason="DevMemoryPair has no attribute types")
def test_stream_advanced_features():
    def test_custom_target():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: A[i, j] + B[i, j], "C")
            D = hcl.compute(C.shape, lambda i, j: C[i, j] + 1, "D")
            return D

        config = {
            "host": hcl.dev.CPU("intel", "e5"),
            "xcel": [hcl.dev.FPGA("xilinx", "xcvu19p")],
        }

        p = hcl.Platform.custom(config)
        s = hcl.create_schedule([A, B], kernel)
        s.to(A, p.xcel.HBM[0])
        s.to(B, p.xcel.HBM[1])
        s.to(kernel.D, p.host)
        p.config(compiler="vitis", mode="debug", backend="vhls")
        code = hcl.build(s, p)
        assert "MAX_HBM_BANKCOUNT" in code

    def test_multiple_device():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "C")
            D = hcl.compute(C.shape, lambda i, j: B[i, j] + 1, "D")
            return hcl.compute(C.shape, lambda i, j: C[i, j] + D[i, j], "E")

        config = {
            "host": hcl.dev.CPU("intel", "e5"),
            "xcel": [
                hcl.dev.FPGA("xilinx", "xcvu19p"),
                hcl.dev.FPGA("xilinx", "xcvu19p"),
            ],
        }

        p = hcl.Platform.custom(config)
        s = hcl.create_schedule([A, B], kernel)
        s.to(A, p.devs[1])
        s.to(B, p.devs[2])
        s.to(kernel.E, p.host)
        s.to(kernel.D, p.host)
        print(hcl.lower(s))

    def test_comm_intf():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: A[i, j] + B[i, j], "C")
            D = hcl.compute(C.shape, lambda i, j: C[i, j] + 1, "D")
            return D

        target = hcl.Platform.aws_f1
        target.config(compiler="vitis", mode="debug")
        s = hcl.create_schedule([A, B], kernel)

        s.to(A, target.xcel, mode=hcl.IO.Stream)
        s.to(B, target.xcel, mode=hcl.IO.DMA)
        s.to(kernel.D, target.host, mode=hcl.IO.Stream)

        code = hcl.build(s, target)
        assert "hls::stream<pkt_b32> &A" in code
        assert "hls::stream<pkt_b32> &D" in code

    def test_stencil_stream():
        hcl.init()
        A = hcl.placeholder((10, 10), "A")

        def stencil(A):
            B = hcl.compute(
                (10, 8), lambda y, x: A[y, x] + A[y, x + 1] + A[y, x + 2], "B"
            )
            C = hcl.compute(
                (8, 8), lambda y, x: B[y, x] + B[y + 1, x] + B[y + 2, x], "C"
            )
            return C

        target = hcl.Platform.aws_f1
        target.config(compiler="vitis", mode="debug", backend="vhls")
        s = hcl.create_schedule([A], stencil)

        # create stencil node
        s[stencil.B].stencil(burst_width=256, unroll_factor=4)
        s[stencil.C].stencil(burst_width=128, unroll_factor=8)

        # compute offloading to FPGA
        s.to(A, target.xcel, mode=hcl.IO.DMA)
        s.to(stencil.C, target.host, mode=hcl.IO.Stream)

        code = hcl.lower(s)
        assert "C[0].write" in str(code)

    test_custom_target()
    test_multiple_device()
    test_comm_intf()
    test_stencil_stream()


@pytest.mark.skip(reason="segfault")
def test_mem_customization():
    def test_array_partition():
        if os.system("which vivado_hls >> /dev/null") != 0:
            return

        hcl.init()
        A = hcl.placeholder((10, 10), "A", dtype=hcl.UInt(8))

        def kernel(A):
            B = hcl.compute(
                A.shape, lambda *args: A[args] + 1, name="B", dtype=hcl.UInt(8)
            )
            return B

        target = hcl.Platform.xilinx_zc706
        target.config(
            compiler="vivado_hls",
            mode="csim",
            project="stream_tests/test_array_partition.prj",
        )
        s = hcl.create_schedule([A], kernel)

        s.to(A, target.xcel)
        s.to(kernel.B, target.host)
        s.partition(A, hcl.Partition.Block, dim=1, factor=2)
        s.partition(kernel.B, hcl.Partition.Block, dim=1, factor=2)

        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10, 10))
        np_B = np.zeros((10, 10))

        hcl_A = hcl.asarray(np_A, dtype=hcl.UInt(8))
        hcl_B = hcl.asarray(np_B, dtype=hcl.UInt(8))

    def test_reuse_blur_x_with_data_placement():
        hcl.init()
        A = hcl.placeholder((10, 10), name="A")

        def kernel(A):
            B = hcl.compute(
                (10, 8), lambda y, x: A[y, x] + A[y, x + 1] + A[y, x + 2], name="B"
            )
            C = hcl.compute((10, 8), lambda y, x: B[y, x], name="C")
            return C

        s = hcl.create_schedule([A], kernel)
        kernel_B = kernel.B
        target = hcl.Platform.xilinx_zc706
        target.config(
            compiler="vivado_hls",
            mode="csim",
            project="stream_tests/test_reuse_blur_x_with_data_placement.prj",
        )

        RB = s.reuse_at(A, s[kernel_B], kernel_B.axis[1])
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)

        f = hcl.build(s, target)

    def test_compute_at_blur_x_with_data_placement():
        hcl.init()
        A = hcl.placeholder((10, 10), name="A")

        def kernel(A):
            B = hcl.compute(
                (10, 8), lambda y, x: A[y, x] + A[y, x + 1] + A[y, x + 2], name="B"
            )
            C = hcl.compute((10, 8), lambda y, x: B[y, x], name="C")
            D = hcl.compute((10, 8), lambda y, x: C[y, x], name="D")
            return D

        s = hcl.create_schedule([A], kernel)
        target = hcl.Platform.xilinx_zc706
        target.config(
            compiler="vivado_hls",
            mode="csim",
            project="stream_tests/test_compute_at_blur_x_with_data_placement.prj",
        )

        s[kernel.B].compute_at(s[kernel.C], kernel.C.axis[1])
        s.to(kernel.C, target.xcel)
        s.to(kernel.D, target.host)

        f = hcl.build(s, target)

    def test_reuse_at_with_data_placement():
        hcl.init()
        A = hcl.placeholder((10, 10), name="A")

        def kernel(A):
            B = hcl.compute((10, 10), lambda y, x: A[y, x], "B")
            C = hcl.compute(
                (10, 8), lambda y, x: B[y, x] + B[y, x + 1] + B[y, x + 2], "C"
            )
            return C

        s = hcl.create_schedule([A], kernel)
        target = hcl.Platform.xilinx_zc706
        target.config(
            compiler="vivado_hls",
            mode="csim",
            project="stream_tests/test_reuse_at_with_data_placement.prj",
        )

        s.to(kernel.B, target.xcel)
        RB = s.reuse_at(kernel.B, s[kernel.C], kernel.C.axis[1])
        s.to(kernel.C, target.host)
        f = hcl.build(s, target)

    test_array_partition()
    test_reuse_blur_x_with_data_placement()
    test_compute_at_blur_x_with_data_placement()
    test_reuse_at_with_data_placement()


@pytest.mark.skip(reason="for_() got an unexpected keyword argument 'name'")
def test_dataflow_primitive():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    target = hcl.Platform.aws_f1

    def kernel(A, B):
        C = hcl.compute((10, 32), lambda *args: 0, "C")
        D = hcl.compute(C.shape, lambda *args: 0, "D")
        with hcl.for_(0, 10, tag="Super") as j:
            hcl.update(D, lambda *args: j * A[args] + B[args], name="update.D")
            hcl.update(C, lambda *args: A[args] + j * D[args], name="update.C")
        return C

    def _test_dataflow_loop_body():
        s = hcl.create_schedule([A, B], kernel)
        s.to([A, B], target.xcel)
        s.to(kernel.Super.C, target.host)
        s[kernel.Super].dataflow(kernel.Super.axis[0])
        code = str(hcl.build(s, target="vhls"))
        assert "#pragma HLS dataflow" in code

    def _test_dataflow_region_in_func():
        s = hcl.create_schedule([A, B], kernel)
        s.to([A, B], target.xcel)
        s.to(kernel.Super.C, target.host)
        top = s.subgraph()[0]
        s[top].dataflow()
        code = str(hcl.build(s, target="vhls"))
        assert "#pragma HLS dataflow" in code, code

    _test_dataflow_loop_body()
    _test_dataflow_region_in_func()


@pytest.mark.skip(reason="segfault")
def test_dataflow_graph():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    C = hcl.placeholder((10, 32), "C")

    def kernel(A, B, C):
        D = hcl.compute(A.shape, lambda y, x: A[y, x] + B[y, x], "D")
        E = hcl.compute(C.shape, lambda y, x: C[y, x] * D[y, x], "E")
        F = hcl.compute((10, 30), lambda y, x: E[y, x] + E[y, x + 1] + E[y, x + 2], "F")
        return F

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls",
        mode="csim",
        project="stream_tests/test_dataflow_graph.prj",
    )
    # E.reuse.partition is atatched to F
    s = hcl.create_schedule([A, B, C], kernel)
    RB = s.reuse_at(kernel.E, s[kernel.F], kernel.F.axis[1])
    s.partition(RB, hcl.Partition.Block)
    s.partition(kernel.D, hcl.Partition.Block)

    # create super stage for sub-graphs
    s.to([A, B, C], target.xcel)
    s.to(kernel.E, target.host)
    f = hcl.build(s, target)
    code = str(hcl.lower(s))
    assert "test(A, B, C, E)" in code, code

    # test VHLS and AOCL codegen
    code = str(hcl.build(s, target="vhls"))
    code = str(hcl.build(s, target="aocl"))
    print("Succeed!")


@pytest.mark.skip(reason="Schedule object has no attribute subgraph")
def test_subgraph():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    C = hcl.placeholder((10, 32), "C")

    def kernel(A, B, C):
        D = hcl.compute(A.shape, lambda y, x: A[y, x] + B[y, x], "D")
        E = hcl.compute(C.shape, lambda y, x: C[y, x] * D[y, x], "E")
        F = hcl.compute((10, 30), lambda y, x: E[y, x] + E[y, x + 1] + E[y, x + 2], "F")
        return F

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls", mode="csim", project="stream_tests/test_subgraph.prj"
    )
    # E.reuse.partition is atatched to F
    s = hcl.create_schedule([A, B, C], kernel)
    RB = s.reuse_at(kernel.E, s[kernel.F], kernel.F.axis[1])
    s.partition(RB, hcl.Partition.Block)
    s.partition(kernel.D, hcl.Partition.Block)

    s.to([A, B, C], target.xcel)
    s.to(kernel.E, target.host)

    # create new sch and return top stage
    tops = s.subgraph()
    for top in tops:
        print(top.op.body)

    print(hcl.lower(s))


@pytest.mark.skip(reason="Expected str, bytes or os.PathLike Object, not NoneType")
def test_sobel_vivado_hls():
    width, height = 224, 224
    A = hcl.placeholder((height, width, 3), "A")
    Gx = hcl.placeholder((3, 3), "Gx")
    Gy = hcl.placeholder((3, 3), "Gy")

    def sobel(A, Gx, Gy):
        B = hcl.compute(
            (height, width), lambda x, y: A[x][y][0] + A[x][y][1] + A[x][y][2], "B"
        )
        r = hcl.reduce_axis(0, 3)
        c = hcl.reduce_axis(0, 3)
        D = hcl.compute(
            (height - 2, width - 2),
            lambda x, y: hcl.sum(B[x + r, y + c] * Gx[r, c], axis=[r, c], name="sum1"),
            "xx",
        )

        t = hcl.reduce_axis(0, 3)
        g = hcl.reduce_axis(0, 3)
        E = hcl.compute(
            (height - 2, width - 2),
            lambda x, y: hcl.sum(B[x + t, y + g] * Gy[t, g], axis=[t, g]),
            "yy",
        )
        return hcl.compute(
            (height - 2, width - 2),
            lambda x, y: hcl.sqrt(D[x][y] * D[x][y] + E[x][y] * E[x][y]) * 0.05891867,
            "Fimg",
        )

    s = hcl.create_schedule([A, Gx, Gy], sobel)
    LBX = s.reuse_at(sobel.B, s[sobel.xx], sobel.xx.axis[0], "LBX")
    LBY = s.reuse_at(sobel.B, s[sobel.yy], sobel.yy.axis[0], "LBY")
    WBX = s.reuse_at(LBX, s[sobel.xx], sobel.xx.axis[1], "WBX")
    WBY = s.reuse_at(LBY, s[sobel.yy], sobel.yy.axis[1], "WBY")
    s.partition(LBX)
    s.partition(LBY)
    s.partition(WBX)
    s.partition(WBY)
    s.partition(Gx)
    s.partition(Gy)
    s[sobel.xx].pipeline(sobel.xx.axis[1])
    s[sobel.yy].pipeline(sobel.yy.axis[1])

    target = hcl.Platform.xilinx_zc706
    s.to([A, Gx, Gy], target.xcel)
    s.to(sobel.Fimg, target.host)

    target.config(compiler="vivado_hls", mode="debug")
    print(hcl.build(s, target))


@pytest.mark.skip(reason="function object has no attribute Super")
def test_super_stage():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    target = hcl.Platform.aws_f1

    def kernel(A, B):
        C = hcl.compute((10, 32), lambda *args: A[args] + B[args], "C")

        # with hcl.Stage("Super") as m:
        hcl.update(C, lambda *args: C[args] + 1, "update")

        #     with hcl.Stage("Plus") as stage:
        with hcl.for_(0, 10) as j:
            C[j, 0] = 10
        return C

    # place the whole super stage body on device
    def _test_super_stage_on_device():
        s = hcl.create_schedule([A, B], kernel)

        s.to([A, B], target.xcel)
        s.to(kernel.Super.Plus.C, target.host)

        code = str(hcl.lower(s))
        assert "test(C, A, B)" in code, code
        print("Succeed!")

    # place the whole super stage body on device
    def _test_super_stage_on_device_stream():
        s = hcl.create_schedule([A, B], kernel)

        s.to([A, B], target.xcel, mode=hcl.IO.Stream, fifo_depth=10)
        s.to(kernel.Super.Plus.C, target.host, fifo_depth=10)
        code = str(hcl.lower(s))
        assert (
            'io attr: "C" mem(0) port(0) io_type(0) fifo_depth(10) direction(2)' in code
        ), code
        print("Succeed!")

    # yet to support
    def _test_partial_super_stage_on_device():
        s = hcl.create_schedule([A, B], kernel)
        s.to([A, B], target.xcel)
        s.to(kernel.Super.update.C, target.host)

    _test_super_stage_on_device()
    _test_super_stage_on_device_stream()


@pytest.mark.skip(reason="AttributeError: function object has no attribute B")
def test_inter_kernel_channels():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    C = hcl.placeholder((10, 32), "C")

    def kernel(A, C):
        B = hcl.compute((10, 32), lambda *args: 0, "B")

        @hcl.def_([(10, 32), (10, 32)])
        def add(A, B):
            hcl.update(B, lambda *args: A[args] + 1)

        @hcl.def_([(10, 32), (10, 32)])
        def mul(B, C):
            hcl.update(C, lambda *args: B[args] * 2)

        add(A, B)
        mul(B, C)

    s = hcl.create_schedule([A, C], kernel)
    s.to(kernel.mul.B, kernel.add.B, fifo_depth=10)
    code = str(hcl.lower(s))
    print(code)


def test_inter_stage_streaming():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, lambda i, j: C[i][j], "D")
        return D

    target = hcl.Platform.aws_f1
    s = hcl.create_schedule([A, B], kernel)
    s.to(kernel.C, s[kernel.D])
    code = str(hcl.lower(s))
    print(code)


def test_one_stage_on_dev():
    hcl.init()
    dtype = hcl.Float()
    M = 64
    K = 64
    N = 64
    A = hcl.placeholder((M, K), "A", dtype=dtype)
    B = hcl.placeholder((K, N), "B", dtype=dtype)
    k = hcl.reduce_axis(0, K)

    def kernel(A, B):
        C = hcl.compute(
            (M, N),
            lambda x, y: hcl.sum(A[x, k] * B[k, y], axis=k, dtype=dtype),
            "C",
            dtype=dtype,
        )
        return C

    target = hcl.Platform.xilinx_zc706
    target.config(
        compiler="vivado_hls",
        mode="csyn",
        project="stream_tests/test_one_stage_on_dev.prj",
    )

    s = hcl.create_schedule([A, B], kernel)
    s.to([A, B], target.xcel)
    s.to(kernel.C, target.host)
    print(hcl.lower(s))


@pytest.mark.skip(reason="index out of range")
def test_auto_move_to_dev():
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, lambda i, j: C[i][j], "D")
        return D

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls",
        mode="csim",
        project="stream_tests/test_auto_move_to_dev.prj",
    )
    s = hcl.create_schedule([A, B], kernel)

    f = hcl.build(s, target)
    print(f.host_src)


def test_vhls_host_dtype():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return

    dtype = hcl.Fixed(16, 12)
    A = hcl.placeholder((10, 32), "A", dtype=dtype)

    def kernel(A):
        B = hcl.compute(A.shape, lambda *args: A[args] + 1, "B", dtype=dtype)
        return B

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls",
        mode="csim",
        project="stream_tests/test_vhls_host_dtype.prj",
    )
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s, target)
    np_A = np.random.randint(10, size=(10, 32))
    np_B = np.zeros((10, 32))

    hcl_A = hcl.asarray(np_A, dtype=hcl.Fixed(16, 12))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Fixed(16, 12))
    f(hcl_A, hcl_B)


def test_vhls_kernel_interface_naming():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return

    dtype = hcl.Float()
    A = hcl.placeholder((10, 32), "A.1", dtype=dtype)

    def kernel(A):
        B = hcl.compute(A.shape, lambda *args: A[args] + 1, "B.1", dtype=dtype)
        return B

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls",
        mode="csim",
        project="stream_tests/test_vhls_kernel_interface_naming.prj",
    )
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s, target)
    np_A = np.random.randint(10, size=(10, 32))
    np_B = np.zeros((10, 32))

    hcl_A = hcl.asarray(np_A, dtype=hcl.Float())
    hcl_B = hcl.asarray(np_B, dtype=hcl.Float())
    f(hcl_A, hcl_B)


def test_inter_stage_consective_streaming():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return

    dtype = hcl.Float()
    A = hcl.placeholder((10, 32), "A", dtype=dtype)

    def kernel(A):
        B = hcl.compute(A.shape, lambda *args: A[args] + 1, "B", dtype=dtype)
        C = hcl.compute(A.shape, lambda *args: B[args] + 1, "C", dtype=dtype)
        D = hcl.compute(A.shape, lambda *args: C[args] + 1, "D", dtype=dtype)
        return D

    target = hcl.Platform.aws_f1
    target.config(
        compiler="vivado_hls",
        mode="csim",
        project="stream_tests/test_inter_stage_consective_streaming.prj",
    )

    s = hcl.create_schedule([A], kernel)
    s.to(A, target.xcel)
    s.to(kernel.D, target.host)

    # inter stage
    s.to(kernel.B, s[kernel.C])
    s.to(kernel.C, s[kernel.D])

    f = hcl.build(s, target)
    np_A = np.random.randint(10, size=(10, 32))
    np_D = np.zeros((10, 32))

    hcl_A = hcl.asarray(np_A, dtype=hcl.Float())
    hcl_D = hcl.asarray(np_D, dtype=hcl.Float())
    f(hcl_A, hcl_D)


@pytest.mark.skip(reason="Cannot find stage A")
def test_host_to_device_stream():
    dtype = hcl.Float()
    A = hcl.placeholder((10, 32), "A", dtype=dtype)

    def kernel(A):
        B = hcl.compute(A.shape, lambda *args: A[args] + 1, "B", dtype=dtype)
        return B

    s = hcl.create_schedule([A], kernel)
    s.to(A, s[kernel.B])


def test_stream_multi_buffer_access():
    def _test_invalid_stream_pattern():
        A = hcl.placeholder((10,), "A")

        def kernel(A):
            B = hcl.compute(A.shape, lambda i: A[i] + 1, "B")
            C = hcl.compute(
                B.shape, lambda i: hcl.select(i < 9, B[i] + B[i + 1], B[i]), "C"
            )
            return C

        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to([A], target.xcel)
        s.to(kernel.C, target.host)
        s.to(kernel.B, s[kernel.C])

        passed = False
        try:
            code = str(hcl.lower(s))
            passed = True
        except:
            assert not passed

    def _test_valid_stream_pattern():
        A = hcl.placeholder((10,), "A")

        def kernel(A):
            B = hcl.compute(A.shape, lambda i: A[i] + 1, "B")
            C = hcl.compute(B.shape, lambda i: hcl.select(i < 9, B[i] + 1, B[i]), "C")
            return C

        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to([A], target.xcel)
        s.to(kernel.C, target.host)
        s.to(kernel.B, s[kernel.C])
        code = str(hcl.lower(s))

    _test_invalid_stream_pattern()
    _test_valid_stream_pattern()


# if __name__ == '__main__':
#     test_dataflow_graph()
#     test_dataflow_primitive()

#     test_stream_advanced_features()
#     test_kernel_duplicate()
#     test_inner_loop_body_placement()
#     test_multiple_subgraph()
#     test_fork_join()
#     test_stream_multi_buffer_access()
#     test_vhls_kernel_interface_naming()
#     test_super_stage()

#     test_host_to_device_stream()
#     test_inter_kernel_channels()
#     test_sobel_vivado_hls()
#     test_subgraph()
#     test_one_stage_on_dev()
#     test_auto_move_to_dev()
#     test_inter_stage_consective_streaming()
#     test_vhls_host_dtype()

#     test_extern_ops()
#     test_stages_one_to_many()
#     test_mixed_stream()
#     test_mem_customization()
