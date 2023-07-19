# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule primitive base."""
# pylint: disable=no-method-argument

from __future__ import annotations
from abc import ABCMeta, abstractmethod

PRIMITIVES = {}
STAGE_PRIMITIVES = {}


def register_primitive():
    """Register a primitive to the schedule."""

    def dectorator(cls):
        if cls.name in PRIMITIVES:
            raise ValueError(f"Primitive {cls.name} already registered")
        if not issubclass(cls, Primitive):
            raise ValueError(f"Class {cls} is not a subclass of Primitive")
        if hasattr(cls, "is_stage_primitive") and cls.is_stage_primitive:
            STAGE_PRIMITIVES[cls.name] = cls
        else:
            PRIMITIVES[cls.name] = cls
        return cls

    return dectorator


class Primitive(metaclass=ABCMeta):
    """A base class of schedule primitives."""

    @property
    @abstractmethod
    def name():
        """The name of the primitive."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def apply(sch, *args, **kwargs):
        """Apply the primitive to the schedule."""
        raise NotImplementedError
