import tvm
import keras
import tvm.relay.frontend as relay_front
import numpy as np
import heterocl as hcl
import hlib
from tvm.relay.expr import Function, Var, Call, Let, If, TupleGetItem, Tuple
from tvm.relay.ty import TensorType, TupleType

# Exploring the AST here

_convert_map = {
    'nn.dense': hlib.nn.dense,
    'nn.relu': hlib.nn.relu,
    'nn.bias_add': hlib.nn.bias_add,
    #    'prelu'                   : 'hlib.nn.prelu',
    'tanh': hlib.math.tanh,
    'sigmoid': hlib.math.sigmoid,
    'nn.softmax': hlib.nn.softmax,
    'exp': hlib.math.exp,
    'log': hlib.math.log,
    'sqrt': hlib.math.sqrt,
    'nn.conv2d': hlib.nn.conv2d,
    'nn.max_pool2d': hlib.nn.max_pool2d,
    'transpose': hlib.nn.transpose,
    'nn.batch_flatten': hlib.nn.flatten,
    'add': hlib.broadcast_add,
    'sub': hlib.broadcast_sub,
    'multiply': hlib.broadcast_mul,
    'split': hlib.nn.split,
    'full': hlib.math.full,
    'full_like': hlib.math.full_like,
    'zeros': hlib.math.zeros,
    'zeros_like': hlib.math.zeros_like,
    'ones': hlib.math.ones,
    'ones_like': hlib.math.ones_like,
}

_attrib = {
    'nn.conv2d': [
        'strides',
        'padding',
        'dilation',
        'groups',
        'channels',
        'kernel_size',
        'data_layout',
        'kernel_layout',
        'out_layout',
        'out_dtype'],
    'nn.conv2d_transpose': [
        'channels',
        'kernel_size',
        'strides',
        'padding',
        'output_padding',
        'dilation',
        'groups',
        'data_layout',
        'kernel_layout',
        'out_layout',
        'out_dtype'],
    'nn.max_pool2d': [
        'pool_size',
        'strides',
        'padding',
        'layout'],
    'nn.avg_pool2d': [
        'pool_size',
        'strides',
        'padding',
        'layout'],
    'transpose': ['axes'],
    'reshape': [
        'newshape',
        'reverse'],
    'squeeze': ['axis'],
    'nn.dense': [
        'units',
        'out_dtype'],
    'nn.softmax': ['axis'],
    'nn.bias_add': ['axis'],
    'sigmoid': [],
    'tanh': [],
    'nn.relu': [],
    'nn.batch_flatten': [],
    'add': [],
    'sub': [],
    'multiply': [],
    'split': [
        'indices_or_sections',
        'axis'],
    'full': [],
    'full_like': [],
    'zeros': [],
    'zeros_like': [],
    'ones': [],
    'ones_like': [],
}


def get_mod(model, shape):
    relay_model = keras.models.load_model(model)
    module, params = frontend.from_keras(relay_model, shape)
    return module, params


def update_if(cur_dict, ap_dict):
    assert type(cur_dict) == type(ap_dict) == dict
    "type must be a dict"
    for key in ap_dict:
        if key not in cur_dict:
            cur_dict[key] = ap_dict[key]
    return cur_dict


def partial_flatten(l):
    _list = []
    for sublist in l:
        if isinstance(sublist, list):
            for item in sublist:
                _list.append(item)
        else:
            _list.append(sublist)
    return _list


def full_flatten(l):
    def _flatten(l):
        for x in l:
            if isinstance(
                    x, (list, tuple)) and not isinstance(
                    x, (str, bytes)):
                for item in _flatten(x):
                    yield item
            else:
                yield x
    return list(_flatten(l))


def fst(l):
    if isinstance(l, list):
        return fst(l[0])
    else:
        return l


def gen_params(type_dict, env):
    args = []
    params = []
    for var in type_dict:
        if (type_dict[var] == Var):
            args.append(var)
            params.append(env[var])
    return args, params


def gen_func(params, env, size):
    args = []
    for var in params:
        args.append(var)

    def func(*args):
        for i in range(size):
            key = "%" + str(i)
            name = env[key][0]
            _func = env[key][1]
            _args = env[key][2]
            _kwargs = env[key][3]
            arg_list = []
            for var in _args:
                if var in params:
                    arg_list.append(var)
                else:
                    arg_list.append(env[var])
            env[key] = _func(*arg_list, **_kwargs)
            print(env[key].name, ":", env[key].shape)
        return env["%" + str(size - 1)]
    return func


def new_gen_func(params, var, type_dict, env, size):
    args = []
    for _var in params:
        args.append(_var)

    def func(*args):
        for item in var:
            if(type_dict[item] == Call):
                name = env[item][0]
                _func = env[item][1]
                _args = env[item][2]
                _kwargs = env[item][3]
                arg_list = []
                for _var in _args:
                    if _var in params:
                        arg_list.append(_var)
                    else:
                        arg_list.append(env[_var])
                env[item] = _func(*arg_list, **_kwargs)
                print(env[item].name, ":", env[item].shape)
        return env[item]
    return func


def model_extent(func, main=False):
    length = 0
    if isinstance(func, Call):
        length = 1
        for arg in func.args:
            if(isinstance(arg, Call)):
                length += model_extent(arg, main)
        if(isinstance(func.op, Function)):
            length = -1
            length += model_extent(func.op, main)
        return length
    elif isinstance(func, Let):
        length += model_extent(func.value, main)
        length += model_extent(func.body, main)
        return length
    elif isinstance(func, Function):
        length += model_extent(func.body, main)
        return length
    if isinstance(func, Tuple):
        return 1  # Anything past this in new scope
    else:
        return 0


def gen_schedule(args, func):
    return hcl.create_schedule(args, func)

# creating relay_to_hcl parser


def relay_parser(model, shape, frontend='keras', dtype=hcl.Float()):
    hcl.init(dtype)
    if frontend == 'keras':
        relay_model = keras.models.load_model(model)
        module, params = relay_front.from_keras(relay_model, shape)
        body = module.functions[module.global_var_map_["main"]]
        place_num = model_extent(body.body, True)
    elif frontend == 'relay':
        body = model
        place_num = model_extent(body, True)
        params = None

    def getType(ty, name):
        if isinstance(ty, TensorType):
            dtype = ty.dtype
            size = []
            for i in ty.shape:
                size.append(i.value)
            return hcl.placeholder(tuple(size), name, dtype)
        elif isinstance(ty, TupleType):
            t_vars = []
            for i in range(len(ty.fields)):
                var_name = name + "_" + str(i)
                t_vars.append(getType(ty.fields[i], var_name))
            return tuple(t_vars)
        else:
            pass

    def parse_rec(node, place):
        if isinstance(node, Function):
            print("Function: ")
            name = "%" + str(place)
            var = [name]
            type_dict = {name: Function}
            env = {}
            temp_var, temp_type, temp_env = parse_rec(node.body, place - 1)
            var.append(temp_var)
            type_dict.update(temp_type)
            env.update(temp_env)
        elif isinstance(node, Var):
            name = node.name_hint
            var = [name]
            type_dict = {name: Var}
            ty = node.type_annotation
            env = {}
            if node.name_hint in shape:
                dtype = ty.dtype
                env[name] = hcl.placeholder(shape[name], name, dtype)
            else:
                env[name] = getType(ty, name)
            print("Var: " + name)
        elif isinstance(node, TupleGetItem):
            index = node.index
            tup = node.tuple_value
            print("Tuple type:", type(tup))
            if isinstance(tup,Var):
                name = "get_" + tup.vid.name_hint
                ty = tup.type_annotation
                var = [name]
                type_dict = {name: TupleGetItem}
                env = {}
                env[name] = (name, getType(ty, name), index)
            elif isinstance(tup,Call):
                if(not hasattr(node.op, "name")):
                    opname = '%' + str(place - 1)
                else:
                    opname = node.op.name
                name = "get_" + opname
                var=[name]
                type_dict = {name: TupleGetItem}
                env = {}
                env[name] = {name, ,index}
            print("TupleGet: " + name)
        elif isinstance(node, Let):
            name = node.var.vid.name_hint
            print("Let: " + name)
            var = [name]
            type_dict = {name: Let}
            env = {}
            args = []
            kwargs = {}
            ty = node.var.type_annotation
            arg_len = 0
            temp_len = 0
            bind_var = getType(ty, name)
            value = node.value
            val_len = model_extent(value)
            temp_var, temp_type, temp_env = parse_rec(value, place - 1)
            if isinstance(value, Var):
                env[name] = (name, bind_var, temp_type,
                             temp_env[fst(temp_var[0])])
            elif isinstance(value, Function):
                env[name] = (name, bind_var, temp_var, temp_type,
                             temp_env)
            elif isinstance(value, Call):
                if not hasattr(value.op, "name"):
                    opname = "%" + str(place - 1)
                else:
                    opname = value.op.name
                var.append(temp_var)
                args.append(temp_env[temp_var[-1]][0])
                temp_len += len(temp_env)
                env.update(temp_env)
                arg_len = len(var) - temp_len
                for i in range(len(args)):
                    if hasattr(args[i], "name"):
                        if(args[i].name in var):
                            env[args[i].name] = args[i]
                if opname in _attrib:
                    for attr in _attrib[opname]:
                        kwargs[attr] = getattr(value.attrs, attr)
                    env[name] = (name,
                                 bind_var,
                                 _convert_map[opname],
                                 tuple(args),
                                 kwargs)
                else:
                    env[name] = (name,
                                 bind_var,
                                 tuple(args))
            type_dict = update_if(type_dict, temp_type)
            temp_var, temp_type, temp_env = parse_rec(
                node.body, place - (val_len))
            var.append(temp_var)
            type_dict = update_if(type_dict, temp_type)
            env.update(temp_env)
        elif isinstance(node, If):
            print("If not instantiated yet")
        elif isinstance(node, Tuple):
            tup_inx = model_extent(node)
            name = "%" + str(place)
            var = [name]
            type_dict = {name: dict}
            env = {}
            tup_type_dict = {}
            tup = []
            tup_env = {}
            inx = 0
            for field in node.fields:
                if isinstance(field, Tuple):
                    inx = inx + 1
                temp_var, temp_type, temp_env = parse_rec(
                    field, tup_inx - inx)  # assumption
                tup.append(temp_var)
                tup_type_dict.update(temp_type)
                tup_env.update(temp_env)
            env[name] = (name, tup, tup_type_dict, tup_env)
            #print("Tuple not instantiated yet")
        elif isinstance(node, Call):
            if(not hasattr(node.op, "name")):
                opname = '%' + str(place - 1)
            else:
                opname = node.op.name
            print("Call: " + opname)
            name = '%' + str(place)
            args = []
            var = []
            type_dict = {name: Call}
            env = {}
            arg_len = 0
            temp_len = 0
            for arg in node.args:
                temp_var, temp_type, temp_env = parse_rec(arg, place - 1)
                if isinstance(arg, Var):
                    var.append(temp_var[0])
                    var = partial_flatten(var)
                    args.append(temp_env[fst(temp_var[0])])
                elif isinstance(arg, Call):
                    var.append(temp_var)
                    var = partial_flatten(var)
                    args.append(temp_env[temp_var[-1]][0])
                    temp_len += len(temp_env)
                    env.update(temp_env)
                type_dict.update(temp_type)
            arg_len = len(var) - temp_len
            var.append(name)
            kwargs = {}
            for i in range(len(args)):
                if hasattr(args[i], "name"):
                    if(args[i].name in var):
                        env[args[i].name] = args[i]
            if opname in _attrib:
                for attr in _attrib[opname]:
                    kwargs[attr] = getattr(node.attrs, attr)
                env[name] = (name, _convert_map[opname], tuple(args), kwargs)
            else:
                env[name] = (name, tuple(args))
            if isinstance(node.op, Function):
                temp_var, temp_type, temp_env = parse_rec(node.op, place - 1)
                var.append(opname)
                type_dict.update({opname: Function})
                env[opname] = (temp_var, temp_type, temp_env)
        return var, type_dict, env
    out_var, out_type, out_env = parse_rec(body, place_num)
    return out_var, out_type, out_env, place_num, params


def get_relay_model(
        model,
        shape={},
        frontend='keras',
        dtype=hcl.Float(),
        in_params=None):
    out_var, out_type, out_env, place_num, params = relay_parser(
        model, shape, frontend)
    out_var = full_flatten(out_var)
    arg, _param = gen_params(out_type, out_env)
    for i in _param:
        print(type(i))
    func = new_gen_func(_param, out_var, out_type, out_env, place_num)
    # func=gen_func(_param,out_env,place_num)
    print(type(func))
    _inputs = []
    if(params is None):
        params = in_params
    for var in params:
        _inputs.append(hcl.asarray(params[var].asnumpy()))
    s = gen_schedule(_param, func)
    return hcl.build(s), _inputs
