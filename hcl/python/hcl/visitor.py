import tvm, numpy, ast, inspect
from mutator import IRMutator

"""A Python AST visitor that constructs Halide IR

Member Variables
----------------
io_dict: contains the user-specified IO
         key = name
         value = {'arg': argument to tvm.ir_pass.MakeAPI}
var_dict: contains all tvm.var
          It can be For loop_var, Allocate buffer_var, etc.
          key = name
          value = {'var': tvm.var,
                   'type': {for, tvm, intermediate},
                   'ast': the ast node for an input var,
                   'min': the min_value for a For index var,
                   'extent': the extent for a For index var,
                   'allocated': whether the var is allocated}
buffer_dict: contains all tvm.placeholder
             It can be an input/output or the result from tvm.compute
             key = name
             value = {'tensor': tvm.placeholder,
                      'buffer': the tvm buffer,
                      'shape': the shape of the buffer in tuple,
                      'ast': the ast node for the placeholder,
                      'type': {input, compute},
                      'allocated': whether the buffer is allocated}
externs_dict: the functions that defined outside
              key = name
              value = ast of the extern func
arg_list: a list that contains the input/output names

Member Functions
----------------
Refer to each function for details

enter(): the main entrance for building a Halide IR

visit_body(): visit a list of statements

visit_Name(): visit an AST Name node
"""

class Visitor(ast.NodeVisitor):
  # the general true condition for Allocate node
  true = tvm.make.UIntImm("uint1", 1)

  def enter(self, ast_root, extern_funcs = [], args = [], dfg_root = None):
    """The entry point for the AST Visitor

    Parameters
    ----------
    ast_root: the ast root to be visited, usually an ast.Module node

    extern_funcs: a list of the plain code of extern functions

    args: a list that contains the name of the io variables

    Returns
    -------
    ir: the constructed Halide IR
    """

    # initialize the member variables
    self.io_dict = {}
    self.var_dict = {}
    self.buffer_dict = {}
    self.externs_dict = {}
    for f in extern_funcs:
      self.externs_dict[f] = ast.parse(extern_funcs[f]).body[0]
    self.arg_list = args
    self.scope = 0
    self.for_count = 0

    assert(isinstance(ast_root, ast.Module))
    assert isinstance(ast_root.body[0], ast.FunctionDef), "Invalide input to mybuild"
    ir = self.visit_body(ast_root.body[0].body)
    """ FOR DEBUG
    a= ir.body.body.body.first.body.rest.body.body.value.b.value
    b= ir.body.body.body.first.body.rest.loop_var
    print (a == b).__bool__()
    """
    print ir
    return ir

  def visit_body(self, nodes):
    """Visit a list of statements in the Python AST"""

    if len(nodes) == 0: # create a dummy statement node
      return tvm.make.Evaluate(1)
    else:
      first = nodes[0]
      rest = nodes[1:]
      has_rest = len(rest) > 0
      if (isinstance(first, ast.For) or isinstance(first, ast.If)): # imperative code block
        ir_first = self.visit(first)
        if has_rest:
          ir_rest = self.visit_body(rest)
          return tvm.make.Block(ir_first, ir_rest)
        else:
          return ir_first
      elif isinstance(first, ast.Assign):
        targets = first.targets
        value = first.value
        if isinstance(value, ast.Call):
          types = self.check_call_type(value)
          if len(types) == 2 and types[0] == "tvm": # tvm functions
            if types[1] == "compute":
              assert len(targets) == 1, "tvm.compute only has one output, instead of " + str(len(targets))
              ir = self.visit(first)
              if has_rest:
                ir = tvm.make.Block(ir, self.visit_body(rest))
              name = targets[0].id
              if name in self.arg_list:
                return ir
              else:
                buffer = self.get_buffer(name)
                assert buffer, "Internal Error: Undeclared buffer " + name
                return tvm.make.Allocate(buffer['buffer'].data, buffer['tensor'].dtype, \
                    buffer['shape'], self.true, ir)
            elif types[1] == "placeholder":
              assert len(targets) == 1, "tvm.placeholder only has one output, instead of " + str(len(targets))
              if has_rest:
                self.visit(first)
                name = targets[0].id
                buffer = self.get_buffer(name)
                if name in self.arg_list:
                  return self.visit_body(rest)
                else:
                  assert buffer, "Internal Error: Undeclared buffer " + name
                  return tvm.make.Allocate(buffer['buffer'].data, buffer['tensor'].dtype, \
                      buffer['shape'], self.true, self.visit_body(rest))
              else: # last statement in the body, useless tvm.placeholder
                return tvm.make.Evaluate(1)
            else: # tvm.var  must be in arg_list
              assert len(targets) == 1, "tvm.var only has one output, instead of " + str(len(targets))
              if has_rest:
                self.visit(first)
                return self.visit_body(rest)
              else: # last statement in the body, useless tvm.var
                return tvm.make.Evaluate(1)
          else: # other function calls
            ir_first = self.visit(value)
            print "here"
            if has_rest:
              ir_rest = self.visit_body(rest)
              return tvm.make.Block(ir_first, ir_rest)
            else:
              return ir_first
        else:
          # intermediate variable
          ir = self.visit(first)
          target, name, _type = self.get_target(targets[0])
          assert target, "Undeclared variable: " + name
          if _type == 'name':
            target['count'] += 1
          if has_rest:
            ir_rest = self.visit_body(rest)
            ir = tvm.make.Block(ir, ir_rest)
          if _type == 'name':
            if target['allocated'] == True:
              return ir
            else:
              target['count'] -= 1
              if target['count'] == 0:
                target['allocated'] == True
                return tvm.make.Allocate(target['var'], target['var'].dtype, [1], self.true, ir)
              else:
                return ir
          else:
            return ir
      elif isinstance(first, ast.Call) or isinstance(first, ast.Expr):
        ir = self.visit(first)
        if isinstance(ir, tuple): # ignore return value
          ir = ir[0]
        if has_rest:
          ir_rest = self.visit_body(rest)
          return tvm.make.Block(ir, ir_rest)
        else:
          return ir
      else:
        # Not yet supported AST nodes: ClassDef, FunctionDef, Return, Print, While, With, Assert
        return self.visit_body(rest)


  """ Statements """

  def visit_FunctionDef(self, node):
    body = node.body
    stmt = None
    ret = None
    if isinstance(body[-1], ast.Return):
      if len(body[0:-1]) > 0:
        stmt = self.visit_body(body[0:-1])
      ret = self.visit(body[-1])
    else:
      stmt = self.visit_body(body)
    return (stmt, ret)

  def visit_Return(self, node):
    return self.visit(node.value)

  def visit_Assign(self, node):
    """Visit targets = value

    Returns
    -------
    Stmt: Store node, tvm.var, tvm.buffer, or tvm.compute IR
    """
    # Currently, we only allow one output target
    target = node.targets[0]
    index = 0
    content = None
    is_tvm = False
    dtype = "float32"


    # Analyze right hand side first
    if isinstance(node.value, ast.Call):
      call = node.value
      call_type = self.check_call_type(call)
      if len(call_type) == 1:
        # External function call. We do not support it right now
        content = self.visit(call)
      else:
        args = call.args
        keywords = call.keywords
        # Currently we only support tvm calls
        if call_type[0] == "tvm":
          is_tvm = True
          if call_type[1] == "var": # tvm.var
            assert isinstance(target, ast.Name), "target of tvm.var must be a name"
            for keyword in keywords: # check every keyword in tvm.var
              if keyword.arg == "dtype":
                dtype = keyword.value.s
              elif keyword.arg == "name":
                pass
              else:
                raise ValueError("Unknown/Unsupported keyowrds to tvm.var: " + str(keyword[0]))
            name = target.id
            tvm_var = tvm.var(name, dtype = dtype)
            var = {'var': tvm_var, 'type': 'tvm', 'allocated': False}
            if name in self.arg_list: # check whether this var belongs to io
              self.io_dict[name] = {'arg': tvm_var}
              var['allocated'] = True
            self.insert_var(name, var)
            content = None
          elif call_type[1] == "placeholder": # tvm.placeholder
            assert isinstance(target, ast.Name), "target of tvm.placeholder must be a name"
            for keyword in keywords: # check every keyword in tvm.var
              if keyword.arg == "dtype":
                dtype = keyword.value.s
              elif keyword.arg == "name":
                pass
              else:
                raise ValueError("Unknown/Unsupported keyowrds to tvm.placeholder: " + str(keyword[0]))
            name = target.id
            shape = self.get_shape(call.args[0])
            placeholder = tvm.placeholder(shape, name = name, dtype = dtype)
            buff = tvm.decl_buffer(placeholder.shape, placeholder.dtype, placeholder.name)
            buffer = {'tensor': placeholder, 'buffer': buff, 'type': 'input', 'ast': node, 'shape': shape, 'allocated': False}
            if name in self.arg_list:
              self.io_dict[name] = {'arg': buff}
              buffer['allocated'] = True
            self.insert_buffer(name, buffer)
            content = None
          elif call_type[1] == "compute":
            name = target.id
            shape = self.get_shape(call.args[0])
            placeholder = tvm.placeholder(shape, name = name, dtype = dtype)
            buff = tvm.decl_buffer(placeholder.shape, placeholder.dtype, placeholder.name)
            buffer = {'tensor': placeholder, 'buffer': buff, 'type': 'compute', 'ast': node, 'shape': shape, 'allocated': False}
            if name in self.arg_list:
              self.io_dict[name] = {'arg': buff}
              buffer['allocated'] = True
            self.insert_buffer(name, buffer)
            lamb = call.args[1]
            assert isinstance(lamb, ast.Lambda), "The second argument to tvm.compute must be a lambda function"
            self.scope += 1
            ret = self.visit(lamb)
            args = lamb.args.args
            if len(shape) == 1:
              var_name = args[0].id
              var = tvm.var(var_name, "int32")
              st = tvm.make.Store(buff.data, ret, var, self.true)
              if not isinstance(ret, tuple):
                ret = self.ReplaceVar(var_name, var).mutate(ret)
                st = tvm.make.Store(buff.data, ret, var, self.true)
                content = tvm.make.For(var, 0, shape[0], 0, 0, st)
              else:
                ret[0] = self.ReplaceVar(var_name, var).mutate(ret[0])
                ret[1] = self.ReplaceVar(var_name, var).mutate(ret[1])
                st = tvm.make.Store(buff.data, ret[1], var, self.true)
                content = tvm.make.For(var, 0, shape[0], 0, 0, tvm.make.Block(ret[0], st))
            else:
              var_name1 = args[0].id
              var_name2 = args[1].id
              var1 = tvm.var(var_name1, "int32")
              var2 = tvm.var(var_name2, "int32")
              if not isinstance(ret, tuple):
                ret = self.ReplaceVar(var_name1, var1).mutate(ret)
                ret = self.ReplaceVar(var_name2, var2).mutate(ret)
                st = tvm.make.Store(buff.data, ret, (var1 * shape[1] + var2), self.true)
                expr = tvm.make.For(var2, 0, shape[1], 0, 0, st)
              else:
                if ret[0] is not None:
                  ret0 = self.ReplaceVar(var_name1, var1).mutate(ret[0])
                  ret0 = self.ReplaceVar(var_name2, var2).mutate(ret0)
                ret1 = self.ReplaceVar(var_name1, var1).mutate(ret[1])
                ret1 = self.ReplaceVar(var_name2, var2).mutate(ret1)
                st = tvm.make.Store(buff.data, ret1, (var1 * shape[1] + var2), self.true)
                if ret[0] is not None:
                  expr = tvm.make.For(var2, 0, shape[1], 0, 0, tvm.make.Block(ret0, st))
                else:
                  expr = tvm.make.For(var2, 0, shape[1], 0, 0, st)
              content = tvm.make.For(var1, 0, shape[0], 0, 0, expr)
              self.scope -= 1
          else:
            raise ValueError("Unkown/Unsupported tvm function: tvm." + call_type[1])
          return content
        else: # if call_type[1] == "tvm"
          raise ValueError("Currently we only support tvm functions")
    else: # if isinstance(node.value, ast.Call)
      content = self.visit(node.value)
    # left hand side
    var, name, _type = self.get_target(target)
    if _type == 'name':
      if var is None:
        var = tvm.var(name, "float32")
        self.insert_var(name, {'var': var, 'type': 'intermediate', 'allocated': False})
      else:
        var = var['var']
    else:
      index = self.visit(target)
      var = var['buffer'].data

    assert (not is_tvm)
    if isinstance(node.value, ast.IfExp):
      then = tvm.make.Store(var, content[1], index)
      orelse = tvm.make.Store(var, content[2], index)
      return tvm.make.IfThenElse(content[0], then, orelse)
    else:
      return tvm.make.Store(var, content, index)

  def visit_For(self, node):
    """Visit for i in range(a, b)

    Returns
    -------
    Stmt: For node
    """
    self.scope += 1
    index = node.target.id
    min_val = node.iter.args[0].n
    max_val = node.iter.args[1].n
    extent = max_val - min_val
    tvm_var = tvm.var(index, "int32")
    var = {'var': tvm_var, 'type': 'for', 'min': min_val, 'extent': extent, 'allocated': True}
    self.insert_var(index, var)
    stmt = self.visit_body(node.body)
    '''
    for_var = tvm.var("for_"+str(self.for_count), "int32")
    stmt = self.ReplaceVar(index, for_var).mutate(stmt)
    self.for_count += 1
    '''
    self.scope -= 1
    return tvm.make.For(tvm_var, min_val, extent, 0, 0, stmt)

  def visit_If(self, node):
    cond = self.visit(node.test)
    self.scope += 1
    body = self.visit_body(node.body)
    self.scope += 1
    orelse = self.visit_body(node.orelse)
    self.scope -= 2

    return tvm.make.IfThenElse(cond, body, orelse)

  def visit_Expr(self, node):
    return self.visit(node.value)

  """ Expressions """

  def visit_BinOp(self, node):
    # TODO: TYPE CHECK
    left = self.visit(node.left)
    right = self.visit(node.right)
    op = node.op
    if isinstance(op, ast.Add):
      return tvm.make.Add(left, right)
    elif isinstance(op, ast.Sub):
      return tvm.make.Sub(left, right)
    elif isinstance(op, ast.Mult):
      return tvm.make.Mul(left, right)
    elif isinstance(op, ast.Div):
      return tvm.make.Div(left, right)
    elif isinstance(op, ast.Mod):
      return tvm.make.Mod(left, right)
    elif isinstance(op, ast.Pow):
      raise ValueError("tvm does not support power operation")
    elif isinstance(op, ast.LShift):
      return tvm.make.Call("float32", "shift_left", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.RShift):
      return tvm.make.Call("float32", "shift_right", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.BitOr):
      return tvm.make.Call("int32", "bitwise_or", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.BitXor):
      return tvm.make.Call("int32", "bitwise_or", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.BitAnd):
      return tvm.make.Call("int32", "bitwise_or", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.FloorDiv):
      raise ValueError("tvm does not support floor division")
    else:
      raise ValueError("Unkown binary operation" + str(op))

  def visit_UnaryOp(self, node):
    operand = self.visit(node.operand)
    op = node.op
    if isinstance(op, ast.Invert):
      return tvm.make.Call("int32", "bitwise_not", [operand], Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.Not):
      raise tvm.make.Not(operand)
    elif isinstance(op, ast.UAdd):
      return operand
    elif isinstance(op, ast.USub):
      return tvm.make.Mul(operand, -1)
    else:
      raise ValueError("Unkown unary operation" + str(op))

  def visit_Lambda(self, node):
    """Visit lambda x, y: expr

    Returns
    -------
    Stmt, Expr
    """
    args = node.args.args
    body = node.body
    for arg in args:
      assert isinstance(arg, ast.Name), "Argument to the lambda function must be a name"
      name = arg.id
      tvm_var = tvm.var(arg.id, dtype = "int32") #must be a for loop index
      var = {'var': tvm_var, 'type': 'for', 'allocated': True}
      self.insert_var(name, var)
    return self.visit(body)

  def visit_IfExp(self, node):
    test = self.visit(node.test)
    body = self.visit(node.body)
    orelse = self.visit(node.orelse)
    return test, body, orelse

  def visit_Compare(self, node):
    assert len(node.ops) == 1, "We only support one compare operator"
    op = node.ops[0]
    left = self.visit(node.left)
    right = self.visit(node.comparators[0])
    if isinstance(op, ast.Eq):
      return tvm.make.EQ(left, right)
    elif isinstance(op, ast.NotEq):
      return tvm.make.NE(left, right)
    elif isinstance(op, ast.Lt):
      return tvm.make.LT(left, right)
    elif isinstance(op, ast.LtE):
      return tvm.make.LE(left, right)
    elif isinstance(op, ast.Gt):
      return tvm.make.GT(left, right)
    elif isinstance(op, ast.GtE):
      return tvm.make.GE(left, right)
    else:
      raise ValueError("Currently we do not support this operation: " + str(type(op)))

  def visit_Call(self, node):
    names = self.check_call_type(node)
    name = '.'.join(names)
    assert name in self.externs_dict, "Unknown external function: " + name
    self.scope += 1
    new_node = self.CallTransformer().enter(self.externs_dict[name], node.args)
    ret = self.visit(new_node)
    self.scope -= 1
    return ret

  def visit_Num(self, node):
    return node.n

  def visit_Subscript(self, node):
    """Visit A[x] or A[x][y]

    Returns
    -------
    Expr: x or x * extent + y
    """
    index = None
    buffer = None
    if isinstance(node.value, ast.Subscript):
      # a 2D array
      var2 = node.slice.value
      if isinstance(var2, ast.Name):
        var2 = self.get_var(var2.id)['var']
      else:
        var2 = var2.n
      var1 = node.value.slice.value
      if isinstance(var1, ast.Name):
        var1 = self.get_var(var1.id)['var']
      else:
        var1 = var1.n
      buffer_name = node.value.value.id
      buffer = self.get_buffer(buffer_name)
      buffer_shape = buffer['shape']
      index = var1 * buffer_shape[1] + var2
    else:
      var = node.slice.value
      if isinstance(var, ast.Name):
        var = self.get_var(var.id)['var']
      else:
        var = var.n
      buffer_name = node.value.id
      buffer = self.get_buffer(buffer_name)
      index = var
    if isinstance(node.ctx, ast.Load):
      # TODO: check data type
      return tvm.make.Load(buffer['tensor'].dtype, buffer['buffer'].data, index, self.true)
    else:
      return index

  def visit_Name(self, node):
    name = node.id
    var = self.get_var(name)
    tvm_var = var['var']
    if var['type'] == 'for':
      return tvm_var
    if isinstance(node.ctx, ast.Load):
      return tvm.make.Load(tvm_var.dtype, tvm_var, 0, self.true)
    else:
      return tvm_var

  """Helper Functions"""

  def check_call_type(self, node):
    """Check the type of ast.Call

    It could be tvm.xxx or just a simple function

    Returns
    -------
    [module_name, function_name] or [function_name]
    """
    assert isinstance(node, ast.Call), "the input should be ast.Call"
    if isinstance(node.func, ast.Attribute):
      return [node.func.value.id, node.func.attr]
    else:
      return [node.func.id]

  def get_shape(self, node):
    """Get the shape from a function argument

    Returns
    -------
    Tuple(shape[0], (shape[1]))
    """
    if isinstance(node, ast.Tuple):
      shapes = node.elts
      if len(shapes) == 1:
        return (shapes[0].n,)
      elif len(shapes) == 2:
        return (shapes[0].n, shapes[1].n)
      else:
        raise ValueError("Currently we only support up to 2D arrays. You have " + str(len(shapes)) + " dimensions.")
    elif isinstance(node, ast.Attribute):
      assert(node.attr == "shape"), "Wrong attribute: " + node.attr + ", must be shape"
      assert(isinstance(node.value, ast.Name)), "Wrong shape with " + str(node.value)
      name = node.value.id
      buffer = self.get_buffer(name)
      return buffer['shape']

  def get_target(self, target):
    name = None
    if isinstance(target, ast.Name):
      name = target.id
      var = self.get_var(name)
      return var, name, 'name'
    else:
      assert isinstance(target, ast.Subscript)
      if isinstance(target.value, ast.Subscript): # 2D
        name = target.value.value.id
      else:                                       # 1D
        name = target.value.id
      buffer = self.get_buffer(name)
      return buffer, name, 'buffer'

  def insert_var(self, name, var):
    var['count'] = 0
    self.var_dict[(name, self.scope)] = var

  def get_var(self, name):
    index = self.scope
    while index >= 0:
      if (name, index) in self.var_dict:
        return self.var_dict[(name, index)]
      else:
        index -= 1
    return None

  def insert_buffer(self, name, buffer):
    self.buffer_dict[(name, self.scope)] = buffer

  def get_buffer(self, name):
    index = self.scope
    while index >= 0:
      if (name, index) in self.buffer_dict:
        return self.buffer_dict[(name, index)]
      else:
        index -= 1
    return None

  def replace_call_with_load(self, node):
    if isinstance(node, tvm.expr.Call):
      if node.call_type == 5:
        exprs = []
        for arg in node.args:
          exprs.append(self.replace_call_with_load(arg))
        call = tvm.make.Call("int32", node.name, exprs, node.call_type, node.func, node.value_index)
        return call
      elif node.call_type == 3:
        buff = self.buffer_dict[node.name]['buffer']
        axis = self.axis
        if len(axis) == 1:
          load = tvm.make.Load(node.dtype, buff.data, axis[0].var)
        else:
          load = tvm.make.Load("int32", buff.data, axis[0].var * axis[1].dom.extent + axis[1].var)
        return load
    elif isinstance(node, tvm.expr.Var):
      var = self.var_dict[node.name]['var']
      return var
    elif isinstance(node, tvm.expr.Sub):
      a = self.replace_call_with_load(node.a)
      b = self.replace_call_with_load(node.b)
      return tvm.make.Sub(a, b)
    elif isinstance(node, tvm.expr.Mul):
      a = self.replace_call_with_load(node.a)
      b = self.replace_call_with_load(node.b)
      return tvm.make.Mul(a, tvm.make.Cast("int32", b))
    elif isinstance(node, tvm.expr.Add):
      a = self.replace_call_with_load(node.a)
      b = self.replace_call_with_load(node.b)
      return tvm.make.Add(a, b)
    else:
      return node

  """Helper Class"""
  class CallTransformer(ast.NodeTransformer):
    def enter(self, node, args_out):
      assert(isinstance(node, ast.FunctionDef))
      args_in = node.args.args
      self.args_in = []
      for arg in args_in:
        assert(isinstance(arg, ast.Name))
        self.args_in.append(arg.id)
      self.args_out = args_out
      body = []
      for b in node.body:
        body.append(self.visit(b))
      return ast.FunctionDef(node.name, node.args, body, node.decorator_list)

    def visit_Name(self, node):
      if node.id in self.args_in:
        arg = self.args_out[self.args_in.index(node.id)]
        if isinstance(arg, ast.Subscript):
          return ast.Subscript(arg.value, arg.slice, node.ctx)
        elif isinstance(arg, ast.Name):
          return ast.Name(arg.id, node.ctx)
        elif isinstance(arg, ast.Num):
          return arg
        else:
          raise ValueError("Unkown argument type to the lambda function: " + str(type(arg)))
      else:
        return node

  class ReplaceVar(IRMutator):
    def __init__(self, name, var):
      self.name = name
      self.var = var

    def mutate_Var(self, node):
      if node.name == self.name:
        return self.var
      else:
        return node


