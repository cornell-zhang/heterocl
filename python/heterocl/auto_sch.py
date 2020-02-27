from .mutator import Mutator
from . import devices, util, api
from .tvm import make as _make
from .tvm import expr as _expr
from .tvm import ir_pass as _pass
from .tvm import tensor
import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def auto_sch(sch, func, target, plot=False):
    """ auto split workload to devices """
    graph, op_map = sch.dataflow_graph() 

    # optimize for different devs
    for dev in [target.host, target.xcel]:
        # locality optimization 
        mutator = Evaluator()
        for node in graph: 
            op = op_map[node].op
            if not isinstance(op, tensor.PlaceholderOp):
                if "reducer" in node: continue
                locality(sch, func, node, op_map, dev, mutator)
        # perfom sub-grouping  
        hypernode = dict() 
        for node in graph: 
            op = op_map[node].op
            if not isinstance(op, tensor.PlaceholderOp):
              # search the connected nodes
              for _ in graph.predecessors(node):
                if isinstance(op_map[_].op, tensor.PlaceholderOp) or \
                   "reducer" in _: continue # skip palceholders
                else: # TODO consider complex merging  
                  grouping(_, node, sch, func, target)
                  hypernode[_] = node
                  
        # succs = [_ for _ in graph.successors(node)]
        # print(hypernode)
        return

    if plot: # colored graph  
        partitions = list()
        for _ in set(sch.placement.values()):
            if "cpu" in str(_): partitions.insert(0, _)
            else: partitions.append(_)
        colors  = ["lightblue", "red"] # cpu & fpga
        mapping = dict(zip(partitions, colors))
        color_map = []
        color = "lightblue"

        # create device color mapping 
        for node in graph:
          if node in sch.placement:
            color = mapping[sch.placement[node.name]]
          color_map.append(color)
    
        pos = nx.nx_pydot.graphviz_layout(graph, prog="fdp")
        nx.draw(graph, pos=pos, font_size=5, 
                with_labels=True, node_color=color_map, 
                edge_color="black", label_pos=0.3)
        plt.show()

# data locality reordering + parallel + splitting 
def locality(sch, func, node, op_map, dev, mutator):
    stmt   = op_map[node].op.body
    _stage = getattr(func, node)
    axis = _stage.axis   # iter vars
    stage  = sch[_stage] # tvm stage
    # collect load expression
    mutator.clear()
    mutator.mutate(stmt)
    ld = mutator.load_map
    st = mutator.store_map
    # dep checks (TODO: check distance)
    dependency = dict()
    for _ in ld.keys():
        if _ in st.keys(): 
            if "reducer" in str(_): continue
            print("dependency: ", _)
            dependency[_] = st[_] - ld[_] 

    # evaluate each load index 
    evaluator = ReuseEval(axis)
    evaluator.mutate(stmt) # record itervar & access pattern
    itervar = {_.var:[(_.dom.min, _.dom.extent)] for _ in axis}
    inputs = {_.name: _.shape for _ in stage.op.input_tensors}

    def recover_index(range_, expr, axis):
        indices = []
        for _ in range(len(range_)-1,-1,-1): # 
            index = util.CastRemover().mutate(_pass.Simplify(expr % range_[_]))
            index = ModRemover(range_[_], axis).mutate(index)
            expr = _pass.Simplify((expr - index) // range_[_])
            expr = util.CastRemover().mutate(expr)
            indices.insert(0, index)
        return indices

    # reusability detection 
    def reuse_check(ivs, shape):
        reuse    = {_.var : [] for _ in ivs}
        parallel = {_.var : [] for _ in ivs}
        min_map = {_.var : _.dom.min.value for _ in ivs}
        max_map = {_.var : _.dom.extent.value-1 for _ in ivs}
        for k, v in ld.items(): # pattern analysis for each load 
            if "reducer" in str(k): continue
            indices = recover_index(inputs[k.name], v[0], max_map)
            if len(indices) > len(ivs): continue # mod index [i%2, i...]
            # check dim index in uniform representation
            for dim in range(len(indices)):
                expr = indices[dim]
                if expr == 0: continue # extent = 1
                m_max = {k:v for k,v in max_map.items() if k.name!=ivs[dim].var.name}
                m_min = {k:v for k,v in min_map.items() if k.name!=ivs[dim].var.name}
                s_max = _pass.Simplify(_pass.Substitute(expr, m_max))
                s_min = _pass.Simplify(_pass.Substitute(expr, m_min))
                next_map = {ivs[dim].var: ivs[dim].var + 1}
                next_min = _pass.Simplify(_pass.Substitute(s_min, next_map)) 
                distance = _pass.Simplify(s_max - next_min).value
             
                if not isinstance(distance, int): continue # mod
                if distance > 0: # found reusibility  
                    if not ivs[dim].var.name in str(expr): # parallelizable
                        parallel[ivs[dim].var].append(k) 
                    # input reusable in dim-th axis
                    elif not isinstance(expr, _expr.Mod):
                        if not "stride_h\"=1 \"stride_w\"=1" in str(stmt): continue
                        reuse[ivs[dim].var].append(k)
                        # print(ivs[dim].var, distance, expr, k, indices)

        return reuse, parallel

    shape = stage.op.output(0).shape
    reuse, para = reuse_check(axis, shape)

    reused = -1 # reuse_at (intra-stage check)
    buf, w_buf = None, None
    pholders = {_.name:_ for _ in sch.inputs}
    for index in range(len(axis)): 
        if len(reuse[axis[index].var]) > 0:
            for input_ in reuse[axis[index].var]:
                if reused == -1:
                    # _ = op_map[input_.name].op.output(0)
                    if hasattr(func, input_.name): # extern op
                        _ = getattr(func, input_.name)._op 
                    else: _ = pholders[input_.name]
                    buf = sch.reuse_at(_, stage, axis[index]) 
                    sch.partition(buf, dim=3, factor=2)
                    # print(reuse, stmt, _, axis[index]); api.lower(sch)
                    reused = index
                elif w_buf is None and reused == index - 1 and buf is not None: 
                    assert isinstance(buf, tensor._Tensor), reused
                    # TODO cannot create window buffer
                    w_buf = sch.reuse_at(buf, stage, axis[index])

    for k, v in itervar.items():
        s, e = v[0] # potential split  
        if e.value - s.value > 1024 and sum(v[1:]) > 1024:
            idx = axis.index(k) 
            xo, xi = stage.split(axis[idx], factor=32)
            stage.reorder(xi, xo)

    bound, threshold = 1, 1024
    for idx in range(len(axis)-1,-1,-1): # parallel & pipeline
        s, e = itervar[axis[idx].var][0]
        bound *= (e.value - s.value) 
        if bound > threshold:
            if bound // threshold > 1 and False: 
                xo, xi = stage.split(axis[idx], factor=bound//threshold)
                stage.parallel(xo); stage.pipeline(xi); break
            else: # pipeline on idx
                stage.parallel(axis[max(0, idx-1)]); 
                stage.pipeline(axis[idx]); break

# memory benefits > redundant compute
def grouping(pred, curr, s, func, target):
    # Merge loop nests
    p = getattr(func, pred)
    c = getattr(func, curr)
    try: # group
      pass

    except Exception as e:
      print(e)
     
# measure the cost with prmtv & stream applied
def cost_model(graph, op_map, target):
    pcie_bw = 16 # host & xcel communication  
    axis_bw = 10 # from local ddr to on-chip memory 
    print(stmt); import sys; sys.exit(0)
    
    cost = 0 # host to global memory communication cost
    for _ in self.placement.keys(): 
      tensor = op_map[_].op.output(0)
      shape = [_.value for _ in tensor.shape] 
      cost += int(''.join(x for x in tensor.dtype if x.isdigit())) * \
              np.prod(np.array(shape)) / pcie_bw / float(8*2**30)

# mod remover (to extract indices)
class ModRemover(Mutator):

    def __init__(self, mod_, range_):
        self.mod_   = mod_
        self.range_ = range_
 
    def mutate_Mod(self, node):
        if isinstance(node.a, _expr.Var):
            diff = _pass.Simplify(_pass.Substitute(node.a, self.range_) <= self.mod_)
            if diff.value == 1: return node.a
            else: return node
        elif isinstance(node.a, _expr.Add):
            v_max = _pass.Simplify(_pass.Substitute(node.a, self.range_)) 
            diff = _pass.Simplify(v_max + 1 <= self.mod_)
            if diff.value == 1: return node.a 
            # a = self.mutate(_pass.Simplify(node.a % self.mod_))
            # b = self.mutate(_pass.Simplify(node.b % self.mod_))
            return node
        # print(type(node.a), node.a)
        return node

# replace var as expr
class ReuseEval(Mutator):

    def __init__(self, itervars):
        self.ivs = [_.var for _ in itervars]
        self.load_map  = dict()
        self.store_map = dict()
 
    def mutate_Var(self, node):
        return node

    def mutate_Load(self, node):
        index = node.index
        if node.buffer_var not in self.load_map:
             self.load_map[node.buffer_var] = []
        self.load_map[node.buffer_var].append(index)
        return node

    def mutate_Store(self, node):
        if node.buffer_var not in self.store_map:
             self.store_map[node.buffer_var] = []
        self.store_map[node.buffer_var].append(node.index)
        return node

    def mutate_For(self, node):
        loop_var = self.mutate(node.loop_var)
        _min = self.mutate(node.min)
        extent = self.mutate(node.extent)
        body = self.mutate(node.body)
        return _make.For(loop_var, _min, extent, node.for_type, node.device_api, body)


class Evaluator(Mutator):
  # evaluate the locality on target 
  def __init__(self, 
               cache_size=256,   # for locality eva
               vector_width=16,  # for vector op 
               load_cost=10):    # load cost
    self.cache     = cache_size
    self.vector    = vector_width
    self.load_cost = load_cost
    self.load_map  = dict()
    self.store_map = dict()

  def clear(self):
    self.load_map  = dict()
    self.store_map = dict()

  def enter(self, ops):
      stmts = []
      for o in ops:
        if o.body is None:
          stmts.append(None)
        else:
          stmts.append(self.mutate(o.body))
      return stmts

  def mutate_Var(self, node):
      return node

  def mutate_Cast(self, node):
      return self.mutate(node.value)

  def mutate_Load(self, node):
      index = node.index
      if node.buffer_var not in self.load_map:
           self.load_map[node.buffer_var] = []
      self.load_map[node.buffer_var].append(index)
      load = _make.Load(node.dtype, node.buffer_var, index, node.predicate)
      return load

  def mutate_Store(self, node):
      dtype = node.value.dtype
      value = self.mutate(node.value)
      index = node.index
      if node.buffer_var not in self.store_map:
           self.store_map[node.buffer_var] = []
      self.store_map[node.buffer_var].append(index)
      return _make.Store(node.buffer_var, _make.Cast(dtype, value), index, node.predicate)

  def mutate_GetBit(self, node):
      a = self.mutate(node.a)
      gb = _make.GetBit(a, node.index)
      if isinstance(a, _expr.Quantize):
        return _make.Quantize(gb, self.dt_var)
      return gb

  def get_bits(self, expr):
    dtype = expr.dtype
    if dtype[0:3] == "int":
      return True, int(dtype[3:])
    elif dtype[0:4] == "uint":
      return False, int(dtype[4:])
