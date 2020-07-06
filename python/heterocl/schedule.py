"""A module for compute scheduling."""
#pylint: disable=too-many-instance-attributes, no-self-use, missing-docstring
import networkx as nx
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
from copy import deepcopy
from .tvm import tensor
from .tvm import make as _make
from .tvm import stmt as _stmt
from .tvm import expr as _expr
from .tvm import api as tvm_api
from .tvm import _api_internal
from .tvm._api_internal import _ExternOp
from .debug import DSLError, APIError
from . import util
from .devices import Device, DevMediaPair 
from itertools import count

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
    _ids = count(0)

    def __init__(self, sch, inputs, name=""):
        self.id = next(self._ids)
        self.sch = sch
        self.inputs = inputs
        self.placement = dict()
        if self.id > 0 and name == "":
            self.name = "s{}".format(self.id)
        else:
            self.name = name

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
            # op_map from string to tensor op
            op_map[name_with_prefix] = self.sch[stage._op]

            if len(name_with_prefix.split('.')) <= level or level == 0:
                for name in names:
                    # insert intermediate stage
                    if name in self.placement.keys():
                        channel, new_stage, dev = self.placement[name]
                        op_map[channel.op.name] = channel
                        graph.add_edge(name, channel.op.name)
                        pos[name] = (level_count[y], y)

                        op_map[new_stage.op.name] = new_stage
                        graph.add_edge(channel.op.name, new_stage.op.name)
                        pos[channel.op.name] = (level_count[y], y)

                        graph.add_edge(new_stage.op.name, name_with_prefix)
                        pos[new_stage.op.name] = (level_count[y], y)
                        if plot:
                            print(name_with_prefix, "<==", new_stage.op.name, "<==", \
                                channel.op.name, "<==", name)

                    elif name.replace("_top.", "") in self.placement.keys():
                        channel, new_stage, dev = self.placement[name.replace("_top.", "")]
                        op_map[channel.op.name] = channel
                        graph.add_edge(name, channel.op.name)
                        pos[name] = (level_count[y], y)

                        op_map[new_stage.op.name] = new_stage
                        graph.add_edge(channel.op.name, new_stage.op.name)
                        pos[channel.op.name] = (level_count[y], y)

                        graph.add_edge(new_stage.op.name, name_with_prefix)
                        pos[new_stage.op.name] = (level_count[y], y)
                        if plot:
                            print(name_with_prefix, "<==", new_stage.op.name, "<==", \
                                channel.op.name, "<==", name)

                    # add children nodes to graph
                    else:
                        if plot:
                            print(name_with_prefix,  " <=== ", name)
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

        if plot: # draw the network
            try:
                from networkx.drawing.nx_agraph import graphviz_layout
            except ImportError:
                raise ImportError("Graphviz and either PyGraphviz or Pydot required")
            pos=graphviz_layout(graph)
            nx.draw(graph, pos, with_labels=True)
            plt.show()

        return graph, op_map


    def subgraph(self, inputs, outputs):
        assert len(inputs) > 0, "empty inputs"
        assert len(outputs) > 0, "empty outputs"

        # check availability
        graph, op_map = self.dataflow_graph()
        inputs  = [ _.name for _ in inputs ]
        outputs = [ _.name for _ in outputs ]

        # from root to parents
        stack = deepcopy(outputs)
        subgraph = list()
        while len(stack) > 0:
            op = stack.pop()
            if op in subgraph: continue
            if op not in outputs:
                subgraph.insert(0, op)
            if op not in graph.nodes:
                op = "_top." + op
            assert op in graph.nodes, \
                "cannot find node " + op + " in " + str(graph.nodes)
            for _ in graph.predecessors(op):
                if not op in inputs:
                    stack.append(_)

        subgraph = OrderedSet(subgraph)
        return subgraph, op_map

    def duplicate(self, inputs, outputs, factor=2):
        """Extract kernel and duplicate the compute unit"""
        subgraph, op_map = self.subgraph(inputs, outputs)
        # combine the stages in subgraph
        for index in range(len(subgraph)):
            if index == len(subgraph) - 1: break;
            pre_stage  = op_map[subgraph[index]]
            post_stage = op_map[subgraph[index+1]]
            axis_num = len(post_stage.op.axis)
            axis = post_stage.op.axis[axis_num-1]
            pre_stage.compute_at(post_stage, axis)

        # split kernel
        post_stage.split(post_stage.op.axis[0], factor)
        return post_stage

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
        except (AttributeError, ValueError):
            try:
                target = target._op
            except AttributeError:
                pass

        if name is None:
            name = target.name + ".reuse"
        return self.sch.reuse_at(target, parent, axis, name)


    def join(self, srcs, dest=None):
        """ join multiple tensors to single dest """
        assert len(srcs) > 0, "joined tensors should be " + \
                "collectde from more than one srcs"

        # create channels and collector stage
        if dest is not None:
            if isinstance(dest, tuple):
                dest, target = dest
                dest = self[dest]
            elif isinstance(dest, Stage):
                target = dest._op
            elif isinstance(dest, tuple):
                src, target = dest
            else: # target tensor
                target = dest.tensor
        else: target = dest

        for src in srcs:
            if isinstance(src, tuple):
                src, tensor = src
                assert tensor == target, + \
                        "inconsistent tensor joining"
            self.sch.join(target, dest, self[src])


    def fork(self, tensor, dests, axis=0):
        """ fork tensor to multiple dests """
        assert len(dests) > 0, "forked tensor should be " + \
                "broadcast to more than one dest"
        # dest as tvm stages
        for dest in dests:
            self.to(tensor, self[dest])


    def to(self, tensors, dst, src=None, axis=0,
           mode=_expr.IO.DMA, depth=1, local_buffer=True, name=None):
        """Stream a list of Tensors to dst devices 
        Parameters
        ----------
        tensors : list of Tensor
            The tensors to be moved

        dst : device or stage
            The destination of data movement

        src : device or stage
            The source of data movement

        axis : axis index
            Move axis-th loop body to xcel scope

        mode : data movement type
            The modes of data movement (FIFO, DMA, MMIO)
            For inter-kernel data movemnet, only FIFO is supported

        depth : channel depth
            The streaming channel depth

        local_buffer : boolean 
            create local buffer for data on-device

        """
        if mode not in [ _expr.IO.DMA, _expr.IO.FIFO ]:
            raise APIError("Invalid channel type")
        rets = list()
        if not isinstance(tensors, list):
            tensors = [tensors]
        for tensor in tensors:
            try:
                if isinstance(tensor, Stage):
                    target = tensor._op
                # unpack tuple of src stage and tensor
                elif isinstance(tensor, tuple):
                    src, target = tensor
                    # from hcl stage to tvm stage
                    src = self.__getitem__(src)
                else: # target tensor
                    target = tensor.tensor
            except (AttributeError, ValueError):
                target = tensor

            # convert hcl stage
            try: dst = self[dst]
            except: pass

            move_to_device = False
            if src is None:
                # move to device
                if isinstance(dst, Device) or \
                        isinstance(dst, DevMediaPair):
                    if axis == 0:
                        move_to_device = True
                    else: # inner-stage movement
                        assert isinstance(tensor, Stage)
                        target = self[tensor]

                else: # inter-stage
                    src = self[tensor]

            # target can be stage or tensor
            ret = self.sch.to(target, dst, src, axis, mode, depth, local_buffer)
            # record the placement information
            if move_to_device:
                channel, ret = ret
                self.placement[target.name] = \
                        (self.__getitem__(channel), \
                         self.__getitem__(ret), dst)

            rets.append(ret)

        if len(rets) == 1: return rets[0]
        else: return rets

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
        self.substages = []
        # Attributes for cross-stage relation
        self.input_stages = set([])
        self.lhs_tensors = set([])
        self.last_substages = set([])
        self.name_with_prefix = self.name if Stage.get_len() == 0 \
                                    else Stage.get_current().name_with_prefix + "." + self.name
        # Private attributes for building a stage
        self._op = None
        self._hcl_dtype = util.get_dtype(dtype, self.name_with_prefix)
        self._dtype = util.get_tvm_dtype(dtype, self.name_with_prefix)
        self._buf = tvm_api.decl_buffer(shape, self._dtype, self.name)
        self._shape = self._buf.shape

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
            # update superstage's substages
            superstage.substages.append(self)
        # Otherwise update the list of stages globally
        else:
            Schedule.stage_ops.append(self)
            Schedule.last_stages.add(self)
            Schedule.last_stages -= self.input_stages

    def __repr__(self):
        return self.name

    def __getattr__(self, name):
        try:
            if name in self.var_dict:
                return self.var_dict[name]
            else:
                # return stage and target tensor op
                for tensor in self.lhs_tensors:
                    if tensor.name == name:
                        return (self, tensor._tensor)
                # check tensors in input stages
                for stage in self.input_stages:
                    if stage.name == name:
                        return (self, stage._op)
                # check tensors in input_stage.lhs
                for stage in self.input_stages:
                    lhs = stage.lhs_tensors
                    for tensor in lhs:
                        if tensor.name == name:
                            return (self, tensor._tensor)
                raise ValueError("Member " + name + \
                    " not found in " + str(self.lhs_tensors) + " or " + \
                    str(self.input_stages))
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
