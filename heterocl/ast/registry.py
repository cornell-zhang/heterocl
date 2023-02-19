# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type inference rule registration."""

from hcl_mlir.exceptions import HCLValueError

TYPE_RULES = {}


def get_type_rules(type_class):
    """Get the type rule."""
    if type_class not in TYPE_RULES:
        raise HCLValueError(
            f"Typing rules not defined for operation type: {type_class}"
        )
    return TYPE_RULES[type_class]


def register_type_rules(type_rule):
    """Register a type rule."""

    def wrapper(cls):
        if cls not in TYPE_RULES:
            TYPE_RULES[cls] = type_rule()
        else:
            raise HCLValueError(f"Type rule for {cls} already exists.")
        return cls

    return wrapper
