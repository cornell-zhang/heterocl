from ._ffi.node import NodeBase, register_node

@register_node
class Location(NodeBase):
    pass

@register_node
class PythonAST(NodeBase):
    pass

@register_node
class PythonASTVar(PythonAST):
    pass

@register_node
class PythonASTAdd(PythonAST):
    pass

