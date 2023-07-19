# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule primitives."""

import os
from os.path import abspath, dirname, join, isfile
from inspect import getsourcefile, getmembers
from importlib import import_module

from .base import register_primitive, PRIMITIVES, Primitive

path = dirname(abspath(getsourcefile(lambda: 0)))
files = [
    f
    for f in os.listdir(path)
    if isfile(join(path, f)) and f not in {"__init__.py", "base.py"}
]
for file in files:
    mod = import_module(f".{file.split('.')[0]}", package="heterocl.primitives")
    # register the schedule primitive using the decorator
    getmembers(mod)
