#!/bin/bash
# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

echo "Check license header..."
python3 scripts/lint/check_license_header.py HEAD~1

echo "Check Python formats using black..."
bash ./scripts/lint/git-black.sh HEAD~1
