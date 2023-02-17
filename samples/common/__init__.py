# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


def innermost(axis):
    return axis[len(axis) - 1]


def unroll_innermost(schedule, tensor, factor=1):
    schedule[tensor].unroll(innermost(tensor.axis), factor=factor)
