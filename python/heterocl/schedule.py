"""A module for compute scheduling."""
#pylint: disable=too-many-instance-attributes, no-self-use, missing-docstring
import networkx as nx
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
from copy import deepcopy
from .tvm import tensor as _tensor
from .tvm import schedule as _schedule
from .tvm import make as _make
from .tvm import stmt as _stmt
from .tvm import expr as _expr
from .tvm import api as tvm_api
from .tvm import _api_internal
from .tvm._api_internal import _ExternOp
from .tvm.schedule import _Stage
from .debug import DSLError, APIError, HCLError
from . import util
from . import types
from .devices import Device, DevMemoryPair, is_mem_onchip
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
    stage_names = set()
    mod_calls = dict()
    last_stages = OrderedSet([])
    _ids = count(0)

    def __init__(self, sch, inputs, outputs, name=""):
        self.id = next(self._ids)
        self.sch = sch
        self.inputs = inputs + outputs
        self.outputs = outputs

        # tensor on hold for chained primitives
        self.cascade_tensor = None
        self.cascade_source = None

        # record the data placement information
        # Example: 
        #   self.placement[tensor_name] = (Stage, device)
        self.placement = dict()

        # record the data stream channels that have been created
        # Example:
        #   self.stream_channels[tensor_name] = [destination stages...]
        self.stream_channels  = dict()

        # dict for op mapping
        self.ops_on_dev  = list()
        self.op_map      = dict()

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


    def subgraph(self):
        inputs, outputs = [], []
        for k, v in self.placement.items():
            stage, dev = v
            if "FPGA" in str(dev): inputs.append(stage)
            else: outputs.append(stage)

        if (len(inputs) == 0) or (len(outputs) == 0):
            raise HCLError("Cannot find subgraph in the CDFG." + \
                " Make sure you move the tensor with .to() before calling .subgraph()")

        # check availability
        graph, op_map = self.dataflow_graph()
        inputs  = [ _.op.name for _ in inputs  ]
        outputs = [ _.op.name for _ in outputs ]

        # from root to parents
        stack = deepcopy(outputs)
        subgraph = list()

        while len(stack) > 0:
            op = stack.pop()
            if op in subgraph: continue
            subgraph.insert(0, op)
            if op not in graph.nodes:
                op = "_top." + op
            assert op in graph.nodes, \
                "cannot find node " + op + " in " + str(graph.nodes)
            for _ in graph.predecessors(op):
                if not op in inputs:
                    stack.append(_)
        subgraph = OrderedSet(subgraph)
        self.ops_on_dev = subgraph
        self.op_map     = op_map

        # Create new self.sch
        self.sch = self.sch.normalize()
        self.sch = _schedule.ScopePartition(self.sch) 
        return self.sch.super_stages

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


    def to(self, tensor, dst=None, src=None, axis=0,
           mode=_expr.IO.DMA, fifo_depth=1, burst_len=-1):

        """Stream a list of Tensors to dst devices 
        Parameters
        ----------
        tensor : Tensor
            The tensor to be moved

        dst : Device or Stage
            The destination of data movement

        src : Device or Stage
            The source of data movement

        axis : str or IterVar
            Move axis-th loop body to xcel scope

        mode : str
            The modes of data movement (Stream, DMA, MMIO)
            For inter-kernel data movemnet, only Stream is supported

        fifo_depth : int
            The streaming channel depth
            We leave an interface here to specify the FIFO depth
            in the future we should be able to infer automatically

        Examples
        --------
        .. code-block:: python

            def kernel(A):
                B = hcl.compute((10,32), lambda *args: A[args], "B")
                C = hcl.compute((10,32), lambda *args: B[args]+1, "C")
                return C
            p = hcl.Platform.xilinx_zc706
                
            # 1. Move tensor A to device
            s.to(A, p.xcel)
            # 2. Move stage B's first loop body to device
            s.to(kernel.B, p.xcel, axis=1)
            # 3. Stream betweem B and C stages
            s.to(kernel.B, kernel.C, fifo_depth=10)
        
            --------

            def kernel(A):
                B = hcl.compute((10,32), lambda *args: 0, "B")
                C = hcl.compute((10,32), lambda *args: 0, "C")

                @hcl.def_()
                def func1(A, B):
                    with hcl.for_(0, 10) as i:
                        with hcl.for_(0, 32) as j:
                            B[i, j] = A[i, j] + 1

                @hcl.def_()
                def func2(B, C):
                    with hcl.for_(0, 10) as i:
                        with hcl.for_(0, 32) as j:
                            C[i, j] = B[i, j] + 1

                func1(A, B)
                func2(B, C)
                return C
            
            # 4. Stream between HCL modules
            s.to(kernel.func1.B, kernel.func2.B)

        """
        if mode not in [ _expr.IO.DMA, _expr.IO.Stream ]:
            raise APIError("Only DMA and Streaming modes are supported...")

        # support chained .to()
        if dst is None:
            dst = tensor
            if self.cascade_tensor is None:
                 raise HCLError("target tensor missing")
            tensor = self.cascade_tensor
            # hold stage has none value when the previous move is to device
            if self.cascade_source is not None:
                src = self.cascade_source

        # handle more than one input tensors for data movement
        # Example: s.to([A, B], p.xcel) 
        if isinstance(tensor, list):
            for t in tensor:
                self.to(t, dst, src, axis, mode, fifo_depth, burst_len)
            return self

        # one-to-many (multi-casting)
        if isinstance(dst, list):
            for d in dst:
                self.to(tensor, d, src, axis, mode, fifo_depth, burst_len)
            return self

        # one-to-one data movement
        # convert hcl stage
        # configuring src
        try: 
            if isinstance(dst, tuple):
               dst, _ = dst 
            dst = self.__getitem__(dst)
        except: 
            pass

        try:
            # move the output tensor of a stage
            if isinstance(tensor, Stage):
                tensor = tensor._op

            # unpack tuple of src stage and tensor
            # E.g. kernel.stage.B = (stage, B)
            elif isinstance(tensor, tuple):
                src, tensor = tensor
                # from heterocl stage to tvm stage
                src = self.__getitem__(src)

            else: # target tensor
                tensor = tensor.tensor

        except (AttributeError, ValueError):
            # if the src is already tvm stage
            if isinstance(tensor, tuple):
                _, tensor = tensor

        move_to_device = False
        if src is None:
            # move to device
            if isinstance(dst, (Device, DevMemoryPair)):
                if axis == 0:
                    move_to_device = True
                else: # inner-stage movement
                    tensor = self.__getitem__(tensor)

            # inter-stage
            # Example: s.to(A, stage) where the stage consumes tensor A
            # in this case, the stage producing the tensor A is src stage
            else: 
                src = self.__getitem__(tensor)

        # inter-stage data movement
        if not (isinstance(dst, Device) or isinstance(dst, DevMemoryPair)):
            # 1. handle inter-HCL-module data streaming 
            if isinstance(src.op, _tensor.PlaceholderOp) and isinstance(dst.op, _tensor.PlaceholderOp): 
                # search the kernel calls globally and find the target tensor
                print("[ INFO ] inter-kernel streaming for target tensor")
                src_stage_name = src.op.name.split(".")[1]
                dst_stage_name = dst.op.name.split(".")[1]
                if src_stage_name not in self.mod_calls:
                    raise HCLError("{} is not called".format(src_stage_name))
                if dst_stage_name not in self.mod_calls:
                    raise HCLError("{} is not called".format(dst_stage_name))
                if len(self.mod_calls[src_stage_name]) > 1:
                    raise HCLError("{} has more than one call sites".format(src_stage_name))
                if len(self.mod_calls[dst_stage_name]) > 1:
                    raise HCLError("{} has more than one call sites".format(dst_stage_name))
                    
                def get_overlap(a, b):
                    return list(set(a) & set(b))[0]

                tensor = get_overlap(self.mod_calls[dst_stage_name][0],
                    self.mod_calls[src_stage_name][0])
                tensor = tensor.tensor

            # 2. check whether the streaming channel has been created 
            # from the target tensor to the destination stage
            dst_stages = set()
            if tensor.name in self.stream_channels.keys():
                dst_stages = self.stream_channels[tensor.name]
            size = len(dst_stages)
            t = (src.op.name, dst.op.name)
            dst_stages.add(t)
            if size == len(dst_stages):
                print("[ Warning ] " + 
                    "the tensor {} has been streamed to stage {}... Ignored"
                    .format(tensor.name, dst.op.name))
                return
            self.stream_channels[tensor.name] = dst_stages

        # save tensor and source stage to support .to chain
        self.cascade_tensor = tensor
        if not move_to_device: self.cascade_source = dst
        else: self.cascade_source = None

        # target can be stage or tensor
        # the pre-processing is finished here
        # run the TVM FFI APIs to annotate the DFG with different modes
        # --------------------------------------------
        ret = None
        # 1. Place a target tensor to device
        if isinstance(dst, (DevMemoryPair, Device)):
            is_pair = not isinstance(dst, Device)
            memory = dst.memory if is_pair else dst.DRAM.memory

            # 1.1 Move a stage's loop body to device
            if isinstance(tensor, _Stage): 
                ret = self.sch.in_stage_move(tensor, dst, src, axis, mode, fifo_depth)

            # 1.2 Move a placeholder or extern op to device 
            else:
                assert isinstance(tensor, _tensor._Tensor), \
                    "input " + str(tensor) + " not a tensor"
                is_private_memory, dev = is_mem_onchip(memory.types)
                dev_port = [dev, memory.channel_id, burst_len]
                if is_private_memory:
                    key = "RAM_{}P_{}".format(memory.port_num, memory.types)
                    dev_port = [dev, key, 0]
                
                # Check if the tensor size is greater than memory capacity
                type_in_bytes = types.get_bitwidth(tensor.dtype) / 8
                tensor_size = type_in_bytes
                for dim in tensor.shape:
                    tensor_size *= int(dim.value)
                tensor_size /= 1024
                if tensor_size > memory.capacity:
                    raise HCLError("Tensor size({} MB) larger than {} memory bank capacity {} MB".\
                        format(tensor_size, memory.types, memory.capacity))

                ret = self.sch.move_to_device(tensor, dst, src, 
                    dev_port, axis, mode, fifo_depth)

        # 2. Inter-stage data movement
        # we need to handle inter-stage or inter-HCL-module separately
        else:
            assert isinstance(dst, _Stage), \
                "dst {} not a stage ".format(str(dst))

            # Collect HCL module information and
            # Check if the target tensor is from HCL module
            hcl_modules = {}
            inter_mod_stream = False
            tgt_tensor_name = tensor.name.split(".")[-1]
            for stage in self.sch.stages:
                if hasattr(stage.op, "body"):
                    if isinstance(stage.op.body, _stmt.KernelDef):
                        mod_name = stage.op.body.name
                        hcl_modules[mod_name] = stage

            # Find dst/src stage and target tensor position index
            # The src/dst are placeholder stages
            # for the ExternOp stages
            if isinstance(dst.op, _tensor.PlaceholderOp) and isinstance(src.op, _tensor.PlaceholderOp):
                inter_mod_stream = True
                print("[ INFO ] performing inter-kernel streaming...")
                shape = [ _.value for _ in tensor.shape ]
                index, dst_match = 0, []

                # Matching the shape and names
                dst_mod_name = dst.op.name.split(".")[1]
                dst_stage = hcl_modules[dst_mod_name]
                for s in dst_stage.op.body.arg_shapes:
                    arg_shape = [ _.value for _ in s ]
                    if shape == arg_shape: 
                        dst_match.append(index)
                    index = index + 1

                assert len(dst_match) > 0, "Stream tensor out of scope"
                if len(dst_match) > 1:
                    names = [ str(n) for n in dst_stage.op.body.args ]
                    expected_arg_name = "_top.{}.{}".\
                        format(dst_mod_name, tgt_tensor_name)
                    assert expected_arg_name in names, "{} {}".\
                        format(expected_arg_name, names)
                    dst_match = [ names.index(expected_arg_name) ]

                # 2.1 Streaming channel between HCL modules 
                if src is not None: 
                    index = 0
                    src_match = []
                    # matching the shape and names
                    src_mod_name = src.op.name.split(".")[1]
                    src_stage = hcl_modules[src_mod_name]
                    for s in src_stage.op.body.arg_shapes:
                        arg_shape = [ _.value for _ in s ]
                        if shape == arg_shape: 
                            src_match.append(index)
                        index = index + 1

                    # Use argument name for matching
                    if len(src_match) > 1: 
                        names = [ str(n) for n in src_stage.op.body.args ]
                        expected_arg_name = "_top.{}.{}".\
                            format(src_mod_name, tgt_tensor_name)
                        assert expected_arg_name in names, "{} {}".\
                            format(expected_arg_name, names)
                        src_match = [ names.index(expected_arg_name) ]

                    axis = []
                    match = [dst_match[0], src_match[0]]
                    ret = self.sch.inter_module_stream(tensor, 
                        dst_stage, src_stage, match, axis, mode, fifo_depth)
                    
                else:
                    # 2.2 Stream from local buffer to HCL module
                    ret = self.sch.local_buffer_to_module_stream(tensor, 
                        dst, src, match, axis, mode, fifo_depth)

            # 2.3. inter-stage FIFO channel
            else: 
                # check if the .to is applied for tensor streaming 
                # or dataflow (PE array) generation. for dataflow generation, 
                # we use the injected information to do operation scheduling 
                # and PE generation during the lowering
                tensor_streaming = True
                if hasattr(src.op, "body"):
                    if isinstance(src.op.body, _stmt.AttrStmt):
                        if src.op.body.attr_key == "kernel_scope":
                            tensor_streaming = False

                if hasattr(dst.op, "body"):
                    if isinstance(dst.op.body, _stmt.AttrStmt):
                        if dst.op.body.attr_key == "kernel_scope":
                            tensor_streaming = False

                if tensor_streaming:
                    if axis != 0:
                        assert len(axis) == 2, "Two axes must have same range"
                        assert axis[0].dom.extent.value == axis[1].dom.extent.value
                    axis = axis if axis != 0 else []
                    # 3.1 Stream from one Stage to another
                    ret = self.sch.inter_stage_stream(tensor, 
                        dst, src, axis, mode, fifo_depth)

                # 3.2 Linking PEs. Inject different types of
                # annotation into the code which is used to 
                # infer the time-space mapping
                else:
                    source_name = "AXI port({})".format(src.op.name) \
                        if isinstance(src.op, _tensor.PlaceholderOp) \
                        else "PE({})".format(src.op.name)
                    print("[ INFO ] Linking {} to PE({}) using {} port...".\
                        format(source_name, dst.op.name, tensor.name))
                    ret = self.sch.__create_inter_pe_channel(tensor, dst, src, fifo_depth)
        # --------------------------------------------
        # record the placement information
        if isinstance(dst, (Device, DevMemoryPair)) and ret is not None:
            self.placement[tensor.name] = (self.__getitem__(ret), dst) 

        return self


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
        # Create non-duplicateing stage names
        while self.name in Schedule.stage_names:
            self.name += "_"
        Schedule.stage_names.add(self.name)

        self.stmt_stack = [[]]
        self.var_dict = {}
        self.axis_list = []
        self.has_break = False
        self.has_return = False
        self.ret_dtype = None
        self.for_level = 0
        self.for_ID = 0
        self.substages = []
        # Attributes for ExternModule
        self.ext_ip_name = None
        self.inputs = []
        self.port_types = []
        self.source = []
        self.command  = []
        # Attributes for cross-stage relation
        self.input_stages = set([])
        self.lhs_tensors = set([])
        self.last_substages = set([])
        self.name_with_prefix = self.name if Stage.get_len() == 0 \
                                    else Stage.get_current().name_with_prefix + "." + self.name
        # Attribute for constant tensor
        self.init_values = None
        self.is_const = False
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
        if self.init_values is not None:
            op = _ExternOp(self.name, "", self.axis_list, input_ops,
                           input_bufs, output_bufs, body,
                           self.init_values, self.is_const)
        else:
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
