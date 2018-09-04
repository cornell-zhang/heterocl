import sys
from .api import *
from .dsl import *
from .types import *
from .nparray import *
from .util import hcl_excepthook
from tvm.intrin import *

sys.excepthook = hcl_excepthook
