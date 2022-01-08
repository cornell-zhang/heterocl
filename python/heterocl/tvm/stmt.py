"""Statement AST Node in TVM.

User do not need to deal with AST node directly.
But they can be helpful for developer to do quick proptyping.
While not displayed in the document and python file.
Each statement node have subfields that can be visited from python side.

.. code-block:: python

    x = tvm.var("n")
    a = tvm.var("array", tvm.handle)
    st = tvm.make.Store(a, x + 1, 1)
    assert isinstance(st, tvm.stmt.Store)
    assert(st.buffer_var == a)
"""
from __future__ import absolute_import as _abs
from ._ffi.node import NodeBase, register_node

class Stmt(NodeBase):
    pass

@register_node
class LetStmt(Stmt):
    pass

@register_node
class AssertStmt(Stmt):
    pass

@register_node
class ProducerConsumer(Stmt):
    pass

@register_node
class For(Stmt):
    Serial = 0
    Parallel = 1
    Vectorized = 2
    Unrolled = 3
    Pipelined = 4

@register_node
class Store(Stmt):
    pass

@register_node
class Provide(Stmt):
    pass

@register_node
class Allocate(Stmt):
    pass

@register_node
class AttrStmt(Stmt):
    pass

@register_node
class Free(Stmt):
    pass

@register_node
class Realize(Stmt):
    pass

@register_node
class Block(Stmt):
    pass

@register_node
class IfThenElse(Stmt):
    pass

@register_node
class Evaluate(Stmt):
    pass

@register_node
class Prefetch(Stmt):
    pass

@register_node
class KernelDef(Stmt):
    pass

@register_node
class KernelStmt(Stmt):
    pass
    
@register_node
class ExternModule(Stmt):
    pass

@register_node
class Return(Stmt):
    pass

@register_node
class Break(Stmt):
    pass

@register_node
class While(Stmt):
    pass

@register_node
class Reuse(Stmt):
    pass

@register_node
class Partition(Stmt):
    Complete = 0
    Block = 1
    Cyclic = 2

@register_node
class Stencil(Stmt):
    pass

@register_node
class StreamStmt(Stmt):
    pass

@register_node
class Print(Stmt):
    pass

@register_node
class MultiBlock(Stmt):
    pass

@register_node
class Assert(Stmt):
    pass