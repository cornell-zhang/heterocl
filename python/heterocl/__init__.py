import sys, os

if "HCLIR" in os.environ:
    ir_name = os.getenv('HCLIR')
else:
    ir_name = "mlir"

IR = ir_name
print("Using {} as IR".format(IR))

if ir_name == "mlir":
    from .mlir.schedule import *
    from .mlir.scheme import *
    from .mlir.build_module import *
    from .mlir.operation import *
    from .mlir.runtime import *
    from .mlir.dsl import *
    from .mlir.intrin import *
    from .mlir.debug import *
    from .types import *
    from .platforms import *
else:
    from .api import *
    from .compute_api import *
    from .types import *
    from .dsl import *
    from .devices import *
    from .platforms import *
    from .nparray import *
    from .debug import hcl_excepthook
    from .tvm.intrin import *
    from .tvm.stmt import Partition
    from .tvm.expr import IO
    sys.excepthook = hcl_excepthook
