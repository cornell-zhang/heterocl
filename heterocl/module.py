# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import copy
from multiprocessing import Process
import numpy as np

from hcl_mlir.dialects import func as func_d
from hcl_mlir.ir import MemRefType
from hcl_mlir.exceptions import APIError, APIWarning, HCLNotImplementedError

from .context import get_context, get_location
from .devices import Platform
from .report import report_stats
from .runtime import execute_fpga_backend, execute_llvm_backend
from .utils import hcl_dtype_to_mlir
from .operation import asarray


class HCLModule:
    def __init__(self, name, src, target, host_src=None, context=None, return_num=0):
        self.name = name
        self.src = src  # device src
        self.host_src = host_src
        self.target = copy.copy(target)
        self.context = context
        self.return_num = return_num

    def run_hls(self, shell=False):
        execute_fpga_backend(self.target, shell)
        report = self.report()
        report.display()

    def __call__(self, *argv):
        if "target" not in self.__dict__:
            raise APIError("No attached target!")
        if "name" not in self.__dict__:
            raise APIError("No module name specified!")
        target = self.target
        if isinstance(target, Platform) and target.tool.name in {
            "vivado_hls",
            "vitis_hls",
        }:
            self.run_hls(shell=True)
        elif target == "llvm":
            # convert python immediate to heterocl tensor
            argv = list(argv)
            for i, arg in enumerate(argv):
                if isinstance(arg, (int, float)):
                    np_array = np.array([arg], dtype=type(arg))
                    argv[i] = asarray(np_array)
            original_results = []
            with get_context(), get_location():
                for op in self.host_src.body.operations:
                    if isinstance(op, func_d.FuncOp) and op.sym_name.value == "top":
                        # check if enough args are provided
                        correct_arg_num = len(op.arguments) + len(op.type.results)
                        if len(argv) != correct_arg_num:
                            raise APIError(
                                f"Incorrect number of arguments provided. Expected {correct_arg_num}, got {len(argv)}."
                            )
                        # test inputs
                        for i, arg in enumerate(op.arguments):
                            if not MemRefType.isinstance(arg.type):
                                continue
                            memref_type = MemRefType(arg.type)
                            assert str(memref_type.element_type) == str(
                                hcl_dtype_to_mlir(argv[i].dtype, signless=True)
                            ), f"Input types: {memref_type.element_type} {hcl_dtype_to_mlir(argv[i].dtype, signless=True)}"
                            if tuple(memref_type.shape) != argv[i].np_array.shape:
                                APIWarning(
                                    f"Shape mismatch between input {tuple(memref_type.shape)} and kernel argument {argv[i].np_array.shape}!"
                                ).warn()
                                pad_shape = []
                                for dst, src in zip(
                                    memref_type.shape, argv[i].np_array.shape
                                ):
                                    pad_shape.append((0, dst - src))
                                argv[i].np_array = np.pad(argv[i].np_array, pad_shape)
                        # test outputs
                        for i, res_type in enumerate(op.type.results):
                            if not MemRefType.isinstance(res_type):
                                continue
                            memref_type = MemRefType(res_type)
                            assert str(memref_type.element_type) == str(
                                hcl_dtype_to_mlir(
                                    argv[len(op.arguments) + i].dtype, signless=True
                                )
                            ), f"Output types: {memref_type.element_type} {hcl_dtype_to_mlir(argv[len(op.arguments) + i].dtype, signless=True)}"
                            if (
                                tuple(memref_type.shape)
                                != argv[len(op.arguments) + i].np_array.shape
                            ):
                                APIWarning(
                                    f"Shape mismatch between output {tuple(memref_type.shape)} and kernel result {argv[len(op.arguments) + i].np_array.shape}!"
                                ).warn()
                                pad_shape = []
                                for dst, src in zip(
                                    memref_type.shape,
                                    argv[len(op.arguments) + i].np_array.shape,
                                ):
                                    pad_shape.append((0, dst - src))
                                original_results.append(
                                    [
                                        argv[len(op.arguments) + i],
                                        argv[len(op.arguments) + i].np_array.shape,
                                    ]
                                )
                                argv[len(op.arguments) + i].np_array = np.pad(
                                    argv[len(op.arguments) + i].np_array, pad_shape
                                )
            execute_llvm_backend(self.src, self.name, self.return_num, *argv)
            for res, shape in original_results:
                slicing = []
                for s in shape:
                    slicing.append(slice(0, s))
                res.np_array = res.np_array[tuple(slicing)]
        else:
            raise HCLNotImplementedError(f"Backend {target} is not implemented")

    def report(self):
        """Get tool report"""
        if "target" not in self.__dict__:
            raise APIError("No attached target!")
        if "name" not in self.__dict__:
            raise APIError("No module name specified!")
        target = self.target
        if target.tool.name == "vivado_hls":
            if "csyn" not in target.tool.mode and target.tool.mode != "debug":
                raise APIError(
                    "Not supported mode {target.tool.mode}. Use csyn mode to retrieve the report instead."
                )
        else:
            raise HCLNotImplementedError("target tool {target.tool.name} not supported")
        return report_stats(target, target.project)


class HCLSuperModule:
    def __init__(self, modules):
        self.modules = modules

    def __call__(self):
        if len(self.modules) > 1:
            pool = []
            for module in self.modules:
                pool.append(Process(target=module.run_hls, args=(False,)))
                pool[-1].start()
            for p in pool:
                p.join()
        else:
            self.modules[0].run_hls(True)
