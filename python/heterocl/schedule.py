"""A module for compute scheduling."""
#pylint: disable=too-many-instance-attributes, no-self-use, missing-docstring
import networkx as nx
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
from .tvm import tensor
from .tvm import make as _make
from .tvm import stmt as _stmt
from .tvm import expr as _expr
from .tvm import api as tvm_api
from .tvm import _api_internal
from .tvm._api_internal import _ExternOp
from .debug import DSLError, APIError
from . import util
from . import api

class Schedule(object):
    """Create a compute schedule.

    This is a wrapper class for :obj:`tvm.schedule._Schedule`.

    Parameters
    ----------
    sch : tvm.schedule._Schedule
        The TVM schedule

    inputs : list of Tensor
        Tensors that are the inputs to the schedule
    """

    stage_ops = []
    last_stages = OrderedSet([])

    def __init__(self, sch, inputs):
        self.sch = sch
        self.inputs = inputs
        self.placement = dict()

    def __getitem__(self, stage):
        try:
            return self.sch[stage._op]
        except AttributeError:
            return self.sch[stage.op]

    def dataflow_graph(self, stages=None, level=0, plot=False):
        """Create a dataflow graph for a given schedule.

        Parameters
        ----------
        stages : list of Stage, optional
            The finals stages in the graph. If not specified, draw all the
            stages

        level : int, optional
            The level of stages to draw. If not specified, draw to the
            inner-most stages

        plot : bool, optional
            Whether draw the graph with ``matplotlib`` or not

        Returns
        -------
        networkx.DiGraph
            A directional graph that describes the dataflow
        """
        graph = nx.DiGraph()
        level_count = [0]
        op_map = dict()
        pos = {}

        def gen_graph(stage, y):
            names = []
            for input_stage in stage.input_stages:
                if len(level_count) == y:
                    level_count.append(0)
                names += gen_graph(input_stage, y+1)
            name_with_prefix = stage.name_with_prefix
            op_map[name_with_prefix] = self.sch[stage._op]
            if len(name_with_prefix.split('.')) <= level or level == 0:
                for name in names:
                    graph.add_edge(name, name_with_prefix)
                    pos[name] = (level_count[y], y)
                    level_count[y] += 1
                return [name_with_prefix]
            return names

        if stages is None:
            stages = Schedule.last_stages
        else:
            if not isinstance(stages, (tuple, list)):
                stages = [stages]

        x = 0
        for stage in stages:
            gen_graph(stage, 1)
            pos[stage.name_with_prefix] = (x, 0)
            x += 1

        partitions = list()
        for _ in set(self.placement.values()):
          if "cpu" in str(_): partitions.insert(0, _)
          else: partitions.append(_)
        colors = ["lightblue", "red"] # cpu & fpga
        mapping = dict(zip(partitions, colors))
        color_map = []
        color = "lightblue"

        # create device color mapping 
        for node in graph:
          if node in self.placement:
            color = mapping[self.placement[node]]
          color_map.append(color)
        
        # evaluate the communication cost
        self.cost_model(graph, op_map)

        if plot: # colored graph  
            pos = nx.nx_pydot.graphviz_layout(graph, prog="fdp")
            nx.draw(graph, pos=pos, font_size=5, 
                    with_labels=True, node_color=color_map, 
                    edge_color="black", label_pos=0.3)
            plt.plot()
            plt.show()

        return graph

    def cost_model(self, graph, op_map):
        import numpy as np
        pcie_bw = 16 # host & xcel communication  
        axis_bw = 10 # from local ddr to on-chip memory 
        stmt = api.build(self, "vhls")
        
        cost = 0 # host to global memory communication cost
        for _ in self.placement.keys(): 
          tensor = op_map[_].op.output(0)
          shape = [_.value for _ in tensor.shape] 
          cost += int(''.join(x for x in tensor.dtype if x.isdigit())) * \
                  np.prod(np.array(shape)) / pcie_bw / float(8*2**30)

    def reuse_at(self, target, parent, axis, name=None):
        """Create a reuse buffer reusing the output of current stage

        This returns a new tensor representing the reuse buffer. A stage
        is also built correspondingly. The new stage will be a sub-stage of
        the parent stage under the specified axis. Thus, the axis must be
        inside the axis list of the parent stage.

        Parameters
        ----------
        target : Tensor
            The tensor whose values will be reused

        parent : Stage
            The stage that reuses the output of the current stage

        axis : IterVar
            The axis that generates the reuse values

        name : string, optional
            The name of the reuse buffer

        Returns
        -------
        Tensor
        """
        try:
            target = target.tensor
        except AttributeError:
            try:
                target = target._op
            except AttributeError:
                pass

        if name is None:
            name = target.name + ".reuse"
        return self.sch.reuse_at(target, parent, axis, name)

    def to(self, tensors, dst, src=None,
           stream_type=_expr.StreamExpr.Channel, depth=1, name=None):
        """Stream a list of Tensors to dst devices 
        
        Parameters
        ----------
        tensors : list of Tensor
            The tensors to be moved

        dst : device or module 
            The tensors to be moved

        stream_type : {FIFO, Channel, Burst}, optional
            The stream type
        """
        if stream_type > 2:
            raise APIError("Invalid channel type")
        rets = []
        if not isinstance(tensors, list):
            tensors = [tensors]
        for tensor in tensors: 
            try:
                target = tensor.tensor
            except (AttributeError, ValueError):
                try:
                    target = tensor._op
                except AttributeError:
                    target = tensor
            if name is None:
                name = target.name + ".stream"
            if src is None: # record placement 
                self.placement[target.name] = dst
            ret = self.sch.to(target, dst, src, 
                              stream_type, depth, name)
            name = None
            rets.append(ret)
        return rets

    def partition(self, target, partition_type=_stmt.Partition.Complete, dim=0, factor=0):
        """Partition a Tensor into smaller Tensors or even registers

        Users can specify the partition type, which includes Complete, Block,
        and Cyclic. The default type is Complete, which means we completely
        partition the specified dimension. If Block is specified, the tensor
        is partitioned into N blocks with equal size. The number N is specified
        by the factor. Otherwise, if Cyclic is specified, the elements of the
        tensor is partition in a cyclic manner. For example, if the factor is
        three, the 1st element will be assigned to the 1st partitioned tensor;
        the 2nd element will be assigned to the 2nd one; and so on. Finally, if
        Complete is specified, the factor will be ignored. If `dim` is set to
        0, it means we partition all dimensions.

        Parameters
        ----------
        target : Tensor
            The tensor to be partitioned

        partition_type : {Complete, Block, Cyclic}, optional
            The partition type

        dim : int, optional
            The dimension to be partitioned

        factor : int, optional
            The partition factor
        """
        if partition_type > 2:
            raise APIError("Invalid partition type")
        if dim < 0:
            raise APIError("Invalid dimension")
        if factor < 0:
            raise APIError("Invalid factor")
        try:
            target = target.tensor
        except (AttributeError, ValueError):
            try:
                target = target._op
            except AttributeError:
                pass
        return self.sch.partition(target, partition_type, dim, factor)

    def reshape(self, target, shape):
        """Reshape a Tensor to a specified new shape

        Parameters
        ----------
        target : Tensor
            The tensor to be reshaped

        shape : tuple of int
            The new shape of the tensor
        """
        try:
            target = target.tensor
        except (AttributeError, ValueError):
            try:
                target = target._op
            except AttributeError:
                pass
        _api_internal._ScheduleReshape(self.sch, target, shape)

class Stage(object):
    """Create a stage in the algorithm.

    Stage is needed when an imperative DSL block is not used within any other
    compute APIs. We can further use the created stage to help us schedule
    the imperative components within it. It can also be used to describe a
    higher level of computation hierarchy. For example, we can wrap several
    compute APIs into a single stage.

    Parameters
    ----------
    name : str, optional
        The name of the Stage

    Attributes
    ----------
    stmt_stack : list of list of Stmt
        Store all statements. There are two levels. The outer level is
        for different scopes of statement. The inner level is for
        different statements

    var_dict : dict(str, _Var)
        A dictionary whose key is the name of the variable
        and the value is the variable itself. This enables users to
        access a variable inside a Stage via a Python attribute

    axis_list : list of IterVar
        A list of axes appeared in this Stage

    has_break : bool
        Set to `True` if there is a `break` statement within the stage

    has_return : bool
        Set to `True` if there is a `return` statement within the stage

    ret_dtype : Type
        The returned data type. Only exists for `heterocl.compute`

    for_level : int
        The level of a loop nest where the current statement is.

    for_id : int
        An index used to label the unnamed axes

    input_stages : set of Stage
        A set of stages that are the input to the Stage

    lhs_tensors : set of Tensor
        The tensors that are updated at the left-hand side

    last_substages : set of Stage
        A set of sub-stages that are last used in the current stage

    name_with_prefix : str
        The full name of the stage. This is used when two stages at different
        levels share the same name

    Examples
    --------
    .. code-block:: python

        A = hcl.placeholder((10,))
        with hcl.Stage():
            A[0] = 5
            with hcl.for_(1, 10) as i:
                A[i] = A[i-1] * 2

    """
    _current = []
    """Store all living `Stage`. The newest is at the end."""

    def __init__(self, name=None, dtype=None, shape=()):
        # Attributes related to a single stage
        self.name = util.get_name("stage", name)
        self.stmt_stack = [[]]
        self.var_dict = {}
        self.axis_list = []
        self.has_break = False
        self.has_return = False
        self.ret_dtype = None
        self.for_level = 0
        self.for_ID = 0
        # Attributes for cross-stage relation
        self.input_stages = set([])
        self.lhs_tensors = set([])
        self.last_substages = set([])
        self.name_with_prefix = self.name if Stage.get_len() == 0 \
                                    else Stage.get_current().name_with_prefix + "." + self.name
        # Private attributes for building a stage
        self._op = None
        self._dtype = util.get_dtype(dtype, self.name_with_prefix)
        self._buf = tvm_api.decl_buffer(shape, self._dtype, self.name)
        self._shape = self._buf.shape
        # additional attributes
        self._module = False
        self._inputs = list()

    def __enter__(self):
        Stage._current.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        # update input_stages: the union of the last substages and original input stages
        # collected in the stage
        self.input_stages = self.last_substages.union(self.input_stages)
        # create the output operation
        input_ops = [i._op for i in self.input_stages]
        input_bufs = [i._buf for i in self.input_stages]
        output_bufs = [self._buf] 
        body = self.pop_stmt()
        Stage._current.pop()
        op = _ExternOp(self.name, "", self.axis_list, input_ops,
                       input_bufs, output_bufs, body)
        self._op = op.output(0)
        # update last_update stages
        # if this stage is a substage of other stages
        if Stage._current:
            superstage = Stage._current[-1]
            # add attribute statement for later stage insertion
            superstage.emit(
                lambda x: _make.AttrStmt(self._buf, "attach_scope",
                                         _make.StringImm(superstage.name), x))
            # update the input stages of the superstage:
            # input_stages = original input stages + current input stages - last substages
            superstage.input_stages = superstage.input_stages.union(self.input_stages)
            superstage.input_stages.difference_update(superstage.last_substages)
            # update the last substages of the superstage:
            # last_substages = original substages + current stage - inputs of current stage
            superstage.last_substages.add(self)
            superstage.last_substages.difference_update(self.input_stages)
            # update lhs_tensors:
            # lhs_tensors = original tensors + lhs tensors of current stage
            superstage.lhs_tensors.update(self.lhs_tensors)
            # update var_dict
            superstage.var_dict[self.name] = self
            # update prefix
            self.name_with_prefix = superstage.name_with_prefix + "." + self.name
        else: # otherwise update the list of stages globally
            Schedule.stage_ops.append(self)
            Schedule.last_stages.add(self)
            Schedule.last_stages -= self.input_stages

    def __repr__(self):
        return self.name

    def __getattr__(self, name):
        try:
            return self.var_dict[name]
        except KeyError:
            raise ValueError("Uknown member " + name + " of " + self.name)

    def emit(self, stmt):
        """Insert statements to the current stage."""
        if self.has_break:
            raise DSLError("Cannot write statements after break")
        self.stmt_stack[-1].append(stmt)

    def replace_else(self, if_stmt, else_stmt):
        """Add an ELSE or ELIF branch to an existing IF or ELIF branch."""
        assert isinstance(if_stmt, _stmt.IfThenElse), "Wrong if statement"
        if isinstance(if_stmt.else_case, _stmt.IfThenElse):
            return _make.IfThenElse(if_stmt.condition, if_stmt.then_case,
                                    self.replace_else(if_stmt.else_case, else_stmt))
        return _make.IfThenElse(if_stmt.condition, if_stmt.then_case, else_stmt)

    def pop_stmt(self):
        """Create a statement from the statements within current stage."""
        stmts = self.stmt_stack.pop()
        if not stmts or callable(stmts[-1]):
            stmts.append(_make.Evaluate(0))
        stmt = stmts[-1]
        for s in reversed(stmts[:-1]):
            if callable(s):
                stmt = s(stmt)
            else:
                assert isinstance(s, _stmt.Stmt)
                stmt = _make.Block(s, stmt)
        return stmt

    @staticmethod
    def get_current():
        """Get the current stage."""
        return Stage._current[-1]

    @staticmethod
    def get_len():
        """Get the level of stages."""
        return len(Stage._current)

    @property
    def axis(self):
        """Get the axes of the stage."""
        return self._op.op.axis
