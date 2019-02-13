"""Tensor and Operation class for computation declaration."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
from ._ffi.node import NodeBase, NodeGeneric, register_node, convert_to_node
from . import _api_internal
from . import make as _make
from . import expr as _expr

itervar_cls = None

@register_node("Tensor")
class _Tensor(NodeBase, _expr.ExprOp):
    """Tensor object, to construct, see function.Tensor"""

    @property
    def ndim(self):
        """Dimension of the tensor."""
        return len(self.shape)

    @property
    def axis(self):
        """Axis of the tensor."""
        return self.__getattr__("axis")

    @property
    def op(self):
        """The corressponding :any:`Operation`."""
        return self.__getattr__("op")

    @property
    def value_index(self):
        """The output value index the tensor corressponds to."""
        return self.__getattr__("value_index")

    @property
    def shape(self):
        """The output shape of the tensor."""
        return self.__getattr__("shape")

    @property
    def name(self):
        op = self.op
        if op.num_outputs == 1:
            return op.name
        return "%s.v%d" % (op.name, self.value_index)


class Operation(NodeBase):
    """Represent an operation that generate a tensor"""
    def output(self, index):
        """Get the index-th output of the operation

        Parameters
        ----------
        index : int
            The index size.

        Returns
        -------
        out : Tensor
            The i-th output.
        """
        return _api_internal._OpGetOutput(self, index)

    @property
    def num_outputs(self):
        """Number of outputs of this op."""
        return _api_internal._OpNumOutputs(self)

    @property
    def input_tensors(self):
        """List of input tensors to this op."""
        return _api_internal._OpInputTensors(self)


@register_node
class PlaceholderOp(Operation):
    """Placeholder operation."""
    pass


@register_node
class ComputeOp(Operation):
    """Compute operation."""
    @property
    def axis(self):
        """Represent axis of IterVar, only defined when it is a ComputeOp"""
        return self.__getattr__("axis")

    @property
    def reduce_axis(self):
        """Represent axis of reductions, only defined when it is a ComputeOp"""
        return self.__getattr__("reduce_axis")


@register_node
class ExternOp(Operation):
    """Extern operation."""
    pass
