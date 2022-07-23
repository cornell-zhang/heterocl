IR = "mlir"
print("Using {} as IR".format(IR))

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
from .mlir.instantiate import *