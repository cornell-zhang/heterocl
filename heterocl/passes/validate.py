# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Passes to check the intermediate layer IR


# TODO: check if-elif-else chains are valid
# The following cases are invalid:
# 1. ElseifOp without ElseIfOp or IfOp in front of it
# 2. ElseOp without ElseIfOp or IfOp in front of it

# For example, the following AST is invalid:
# [
#  else if (cond1) {
#     ...
#  }
#  if (cond2) {
#     ...
#  }
#  ]
