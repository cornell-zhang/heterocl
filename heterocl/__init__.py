# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

IR = "mlir"
print("Using {} as IR".format(IR))

import hcl_mlir
from hcl_mlir.ir import *

print("Done HCL-MLIR initialization")

from .schedule import *
from .scheme import *
from .build_module import *
from .operation import *
from .runtime import *
from .dsl import *
from .intrin import *
from .debug import *
from .print import *
from .types import *
from .platforms import *
from .instantiate import *
