# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    parser.addoption("--vhls", action="store", default=False)


@pytest.fixture
def vhls(request):
    return request.config.getoption("--vhls")
