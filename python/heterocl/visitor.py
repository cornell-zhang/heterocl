import tvm, numpy, ast, inspect, re, textwrap
import warnings as wrn
from mutator import IRMutator

def preprocess_source(src, lam = True):
  # remove comments
  src = re.sub(r'#.*\n', "\n",  src)
  src = re.sub(r'\'\'\'.*\'\'\'', "\n", src, flags=re.S)
  # remove header indentations
  src = textwrap.dedent(src)
  # remove trailing comma
  if lam:
    src = src.strip(' \n')
    src = src.rstrip(',\n')
    if src[0] != '(':
      src = src.rstrip(')')
    _src = ''
    lam = 0
    body = False
    scope = 0
    for i in range(0, len(src)):
      if lam == 0 and i+6 < len(src) and src[i:i+6] == 'lambda':
        _src += src[i]
        lam = 1
      elif lam == 1:
        if body == False:
          if src[i] == ":":
            body = True
          _src += src[i]
        elif body == True:
          if scope == 0:
            if src[i] == ",":
              lam = 2
            elif src[i] == "(":
              scope += 1
              _src += src[i]
            else:
              _src += src[i]
          else:
            if src[i] == "(":
              scope += 1
            elif src[i] == ")":
              scope -= 1
            _src += src[i]
    src = _src
  return src

def process_func(_funcs):
  funcs = {}
  for f in _funcs:
    name = f.__name__
    funcs[name] = preprocess_source(inspect.getsource(f), lam = False)
  return funcs

"""A Python AST visitor to extract lambda function"""

class LambdaVisitor(ast.NodeVisitor):
  def enter(self, src):
    src = preprocess_source(src)
    ast_root = ast.parse(src)
    assert isinstance(ast_root, ast.Module)
    return self.visit(ast_root.body[0])

  """ stmt """
  def visit_Tuple(self, node):
    return self.visit(node.elts)

  def visit_ListComp(self, node):
    return None

  def visit_Assign(self, node):
    return self.visit(node.value)

  """ expr """
  def visit_Expr(self, node):
    return self.visit(node.value)

  def visit_Call(self, node):
    for arg in node.args:
      res = self.visit(arg)
      if not res is None:
        return res

  def visit_Lambda(self, node):
    return node

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

scope: record the current scope that will be assigned to variables

Member Functions
----------------
Refer to each function for details

enter(): the main entrance for building a Halide IR

visit_body(): visit a list of statements

visit_Name(): visit an AST Name node
"""

class HalideIRVisitor(ast.NodeVisitor):
  # the general true condition for Allocate node
  true = tvm.make.UIntImm("uint1", 1)

  def compile_lambda(self, ast_root, input_tensors, input_buffers, input_vars, output, extern_funcs):
    """
    outputs = compile ast_root with the given inputs and extern_funcs
    the ast msut be a lambda function
    1. create a block, with the lambda body being compiled and the last one is an assignment: output = ...
    2. compile RHS and get the return value
    3. create a for loop
    """
    self.scope = 0
    self.buffer_dict = {}
    for i, i_b in zip(input_tensors, input_buffers):
      self.buffer_dict[(i_b.name, 0)] = {'tensor': i, 'buffer': i_b, 'shape': i.shape, 'allocated': True}
    self.var_dict = {}
    for i in input_vars:
      self.insert_var(i.name, {'var': i, 'type': 'tvm', 'allocated': True})
    self.externs_dict = {}
    extern_funcs = process_func(extern_funcs)
    for f in extern_funcs:
      self.externs_dict[f] = ast.parse(extern_funcs[f]).body[0]
    assert isinstance(ast_root, ast.Lambda), "Input to HalideIRVisitor must be a lambda function AST"
    ret, indices = self.visit(ast_root)
    shape = output.shape
    body = None
    stmt = None
    index = 0
    assert (len(shape) == len(indices)), "Output dimension must be the same as the nubmer of lambda indices"
    dim = len(indices)
    if dim == 1:
      index = indices[0]
    elif dim == 2:
      index = indices[0] * shape[1] + indices[1]
    if isinstance(ret, tuple):
      stmt = tvm.make.Cast(output.dtype, ret[1])
      if ret[0] is None:
        stmt = tvm.make.Store(output.data, stmt, index, self.true)
      else:
        stmt = tvm.make.Block(ret[0],
            tvm.make.Store(output.data, stmt, index, self.true))
    else:
      stmt = tvm.make.Store(output.data, tvm.make.Cast(output.dtype, ret), index, self.true)
    if dim == 1:
      body = tvm.make.For(indices[0], 0, shape[0], 0, 0, stmt)
    elif dim == 2:
      body = tvm.make.For(indices[0], 0, shape[0], 0, 0,
          tvm.make.For(indices[1], 0, shape[1], 0, 0, stmt))

    return body

  def compile_mut_lambda(self, ast_root, input_tensors, input_buffers, input_vars, shape, extern_funcs):
    self.scope = 0
    self.buffer_dict = {}
    for i, i_b in zip(input_tensors, input_buffers):
      self.buffer_dict[(i_b.name, 0)] = {'tensor': i, 'buffer': i_b, 'shape': i.shape, 'allocated': True}
    self.var_dict = {}
    for i in input_vars:
      self.insert_var(i.name, {'var': i, 'type': 'tvm', 'allocated': True})
    self.externs_dict = {}
    extern_funcs = process_func(extern_funcs)
    for f in extern_funcs:
      self.externs_dict[f] = ast.parse(extern_funcs[f]).body[0]
    assert isinstance(ast_root, ast.Lambda), "Input to HalideIRVisitor must be a lambda function AST"
    ret, indices = self.visit(ast_root)
    body = None
    stmt = None
    index = 0
    assert (len(shape) == len(indices)), "Output dimension must be the same as the nubmer of lambda indices"
    dim = len(indices)
    assert (ret[1] is None), "Lambda function should not return any value"
    if dim == 1:
      body = tvm.make.For(indices[0], 0, shape[0], 0, 0, ret[0])
    elif dim == 2:
      body = tvm.make.For(indices[0], 0, shape[0], 0, 0,
          tvm.make.For(indices[1], 0, shape[1], 0, 0, ret[0]))

    return body

  def compile_block(self, fcompute, inputs, input_buffers, args, extern_funcs):
    self.buffer_dict = {}
    for i, i_b in zip(inputs, input_buffers):
      self.buffer_dict[(i_b.name, 0)] = {'tensor': i, 'buffer': i_b, 'shape': i.shape, 'allocated': True}
    self.var_dict = {}
    fcompute = list(process_func([fcompute]).values())[0]
    ast_root = ast.parse(fcompute).body[0]
    self.externs_dict = {}
    extern_funcs = process_func(extern_funcs)
    for f in extern_funcs:
      self.externs_dict[f] = ast.parse(extern_funcs[f]).body[0]
    self.scope = 0
    self.arg_list = []
    args = self.transform_args(args)
    ret = self.visit_call_with_args(ast_root, args)
    assert isinstance(ret, tuple)
    return ret[0]

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
    buff = self.get_buffer("knn_mat")['tensor']
    s = tvm.create_schedule(buff.op)
    a0 = self.get_var('input_image')['var']
    a1 = self.get_buffer('labelval')['tensor']
    a2 = self.get_buffer('knn_mat')['tensor']
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
    """Visit Function Definition

    Returns
    =======
    (stmt, ret): stmt: the body of function definition, coulf be None
                 ret:  the return expression, could be None
    """
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
            ret = self.visit(lamb)[0]
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
        if isinstance(content, int):
          var = tvm.var(name, "int32")
        elif isinstance(content, tvm.expr.Load):
          var = tvm.var(name, content.dtype)
        else:
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
    """Visit If statement

    Returns
    =======
    Stmt: IfThenElse node
    """
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
      return tvm.make.Call(left.dtype, "shift_left", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.RShift):
      return tvm.make.Call(left.dtype, "shift_right", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.BitOr):
      return tvm.make.Call(left.dtype, "bitwise_or", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.BitXor):
      return tvm.make.Call(left.dtype, "bitwise_xor", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
    elif isinstance(op, ast.BitAnd):
      return tvm.make.Call(left.dtype, "bitwise_and", [left, right], tvm.expr.Call.PureIntrinsic, None, 0)
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
    indices = []
    for arg in args:
      assert isinstance(arg, ast.Name), "Argument to the lambda function must be a name"
      name = arg.id
      tvm_var = tvm.var(arg.id, dtype = "int32") #must be a for loop index
      var = {'var': tvm_var, 'type': 'for', 'allocated': True}
      self.insert_var(name, var)
      indices.append(tvm_var)
    return self.visit(body), indices

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

  def visit_call_with_args(self, node, args):
    self.scope += 1
    node = self.CallTransformer().enter(node, args)
    ret = self.visit(node)
    self.scope -= 1
    return ret

  def visit_Num(self, node):
    return node.n

  def visit_Subscript(self, node):
    """Visit A[x] or A[x][y]
    If the subscript is on rhs, return a Load node
    Otherwise it returns an index

    Returns
    -------
    Expr: x or x * extent + y or Load node
    """
    var, _, index, _ = self.get_index(node, 1)
    if isinstance(node.ctx, ast.Load):
      if isinstance(index, tuple):
        return tvm.make.GetBit(tvm.make.Load(var[0], var[1], index[0], self.true), index[1])
      else:
        return tvm.make.Load(var[0], var[1], index, self.true)
    else:
      return index

  def visit_Name(self, node):
    name = node.id
    var = self.get_var(name)
    tvm_var = var['var']
    if var['type'] == 'for':
      return tvm_var
    elif var['type'] == 'intermediate':
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

  def transform_args(self, args): # TODO: not all cases are considered
    ret = []
    for arg in args:
      if isinstance(arg, tvm.tensor.Tensor):
        ret.append(ast.Name(arg.name, ast.Load))
    return ret

  def get_index(self, node, level):
    var = None
    shape = None
    idx = 0
    lvl = 0
    if isinstance(node.value, ast.Name):
      name = node.value.id
      var = self.get_var(name)
      if var is None:
        var = self.get_buffer(name)
        if var is None:
          raise ValueError("Undefined variable/buffer: " + name)
        else:
          shape = var['shape']
          var = (var['buffer'].dtype, var['buffer'].data)
      else:
        var = (var['var'].dtype, var['var'])
        shape = ()
      if level > len(shape) + 1:
        raise ValueError("Inconsistant dimension access for " + name)
      elif level == len(shape) + 1: #TODO remove this warning
        wrn.warn("Attempting to perform bit operation on " + name, Warning)
      lvl = level
    elif isinstance(node.value, ast.Subscript):
      var, shape, idx, lvl = self.get_index(node.value, level+1)
    else:
      raise ValueError("Wrong input to get_index")
    v = None
    if isinstance(node.slice, ast.Slice):
      return 0
      #TODO: finish this
    elif isinstance(node.slice, ast.Index):
      v = self.visit(node.slice.value)
      assert (v is not None), "Unknown slice value for " + var.name + " at dimension " + str(level)
    else:
      raise ValueError("Unsupported slice type for " + var.name + " at dimension " + str(level))
    if lvl > len(shape):
      if level == 1:
        return var, shape, (idx, v), lvl
      elif level == 2:
        return var, shape, (idx + v), lvl
      else:
        idx = shape[lvl-level+1] * (idx + v)
        return var, shape, idx, lvl
    else:
      if level == 1:
        return var, shape, (idx + v), lvl
      else:
        idx = shape[lvl-level+1] * (idx + v)
        return var, shape, idx, lvl

  """Helper Class"""

  class CallTransformer(ast.NodeTransformer):
    def enter(self, node, args_out):
      assert(isinstance(node, ast.FunctionDef))
      args_in = node.args.args
      assert (len(args_in) == len(args_out)), "Inconsistant arg number of " + str(node.name) + ": expect " + str(len(args_in)) + " but get " + str(len(args_out))
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


