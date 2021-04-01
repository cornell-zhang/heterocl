from .tvm import make as _make
from .tvm import expr as _expr
from .tvm.expr import Var, Call
from .tvm.api import _IterVar, decl_buffer
from . import types
from . import devices
from . import util
import copy
from .debug import DTypeError
from .mutator import Mutator
import heterocl.tvm as tvm

class SubstituteTensor(Mutator):
    def __init__(self, name, var):
      self.name = name
      self.var = var
    def mutate_Var(self, node):
      if node.name == self.name:
        return self.var
      else:
        return node

def substitute(body, vmap):
    for k, v in vmap.items():
        body = SubstituteTensor(k.name, v).mutate(body)
        body = simplify(util.CastRemover().mutate(body))
    return body

def simplify(expr):
    expr = tvm.ir_pass.Simplify(expr) if isinstance(expr, tvm.expr.Expr) else expr
    return util.CastRemover().mutate(expr)

def extract_indices(index, shape, range_):
    new_index = list()
    for i in range(len(shape)-1, 0, -1):
        s = shape[i]
        simple_index = simplify(index % s)
        if isinstance(simple_index, _expr.Mod):
            max_ = simplify(substitute(simple_index.a, range_) + 1)
            comp = simplify(max_ <= simple_index.b)
            if comp == 1:
                simple_index = simple_index.a
        new_index.append(simple_index)
        index = simplify((index - simple_index) / s)
    new_index.append(index)
    reversed_index = [ _ for _ in reversed(new_index) ]
    return reversed_index

# Extract memory access patterns
class AccessPatternAnalyzer(Mutator):
    def __init__(self, target):
        self.itervars = dict()
        self.target = target
        self.access = dict()

    def mutate_ConstExpr(self, node):
        if isinstance(node, _expr.StringImm):
            return node
        return node.value

    def mutate_BinOp(self, binop, node):
        a = self.mutate(node.a)
        b = self.mutate(node.b)
        if isinstance(a, _expr.ConstExpr):
            a = a.value
        if isinstance(b, _expr.ConstExpr):
            b = b.value
        return binop(a, b, False)

    def mutate_Cast(self, node):
        return self.mutate(node.value)
    
    def mutate_For(self, node):
        loop_var = self.mutate(node.loop_var)
        _min = self.mutate(node.min)
        extent = self.mutate(node.extent)
        # record loop itervars
        self.itervars.append(node)
        body = self.mutate(node.body)
        self.itervars.remove(node)
        return _make.For(loop_var, _min, extent, node.for_type, node.device_api, body)

    def mutate_Store(self, node):
        buffer_var = self.mutate(node.buffer_var)
        index = self.mutate(node.index)
        value = self.mutate(node.value)
        predicate = self.mutate(node.predicate)
        varname = buffer_var.name
        if varname == self.target.name:
            # print(f"[ INFO ] Found {varname} writing at index ({index})")
            if "st" not in self.access:
                self.access["st"] = list()
            pattern = [ index, self.itervars.copy() ]
            self.access["st"].append(pattern)
        return _make.Store(buffer_var, value, index, predicate)

    def mutate_Load(self, node):
        buffer_var = self.mutate(node.buffer_var)
        index = self.mutate(node.index)
        predicate = self.mutate(node.predicate)
        varname = buffer_var.name
        if varname == self.target.name:
            # print(f"[ INFO ] Found {varname} reading at index ({index})")
            if "ld" not in self.access:
                self.access["ld"] = list()
            pattern = [ index, self.itervars.copy() ]
            self.access["ld"].append(pattern)
        return _make.Load(node.dtype, buffer_var, index, predicate)

    def analyze(self, target, body, stage_type):
        self.itervars = list()
        self.target = target
        # print(f"[ INFO ] Target tensor shape {target.shape}")
        self.access = dict()
        self.mutate(body)

        # Check the streaming legality
        if len(self.access) > 1:
            raise RuntimeError(f"Target being read/written in the same stage")
        if len(self.access) == 0:
            raise RuntimeError(f"Target tensor not found in the input stage")
        
        for access_type, access in self.access.items():
            if stage_type == "source":
                assert access_type == "st"
            else:
                assert access_type == "ld"
            # Extract multi-dimensional index
            index, itervars = access[0]
            range_ = dict()
            for node in itervars:
                range_[node.loop_var] = simplify(node.extent - 1)
            new_index = extract_indices(index, target.shape, range_)
            # print(f"[ INFO ] {new_index}, {range_}")
            return new_index, range_


# Extract access pattern and check reusibility
def create_reuse_buffers(sch, target, src, dst):
    print(f"[  INFO  ] Analyzing access pattern of {target.name} from {src} to {dst}")
    # print(dst.op.body)
    analyzer = AccessPatternAnalyzer(target)

    # Extract producer(consumer) data writing(reading) pattern
    try:
        wr_index, wr_loop_range_ = analyzer.analyze(target, src.op.body, "source")
        rd_index, rd_loop_range_ = analyzer.analyze(target, dst.op.body, "dest")

        def has_reuse_pattern(index, range_):
            return False

        # TODO: cover more reduction patterns
        # If the consumer's index = [(ra3 + nn), 0, (yy + ra4), (xx + ra5)]
        # then we would have reuse pattern on yy and xx axes
        reuse_axes = list()
        if len(rd_loop_range_) > len(target.shape):
            for i in range(len(rd_index)):
                # If there are two non-zero-extent loop vars in same dimension
                if has_reuse_pattern(rd_index[i], rd_index):
                    reuse_axes.append(i)

        if len(reuse_axes) == 2:
            LB_axis, WB_axis = reuse_axes
            LB = sch.reuse_at(src, dst, dst.axis[LB_axis], "LB")
            WB = sch.reuse_at(LB, dst, dst.axis[WB_axis], "WB")
    except:
        pass

