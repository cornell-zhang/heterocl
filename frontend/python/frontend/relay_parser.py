import tvm,keras
import tvm.relay.frontend as relay_front
import numpy as np
import heterocl as hcl
import hlib
from tvm.relay.expr import Function,Var,Call,Let,If,TupleGetItem
from tvm.relay.ty import TensorType,TupleType

#Exploring the AST here

_convert_map = {
    'nn.dense'                : hlib.nn.dense,
    'nn.relu'                 : hlib.nn.relu,
    'nn.bias_add'             : hlib.nn.bias_add,
#    'prelu'                   : 'hlib.nn.prelu',
    'tanh'                    : hlib.math.tanh,
    'sigmoid'                 : hlib.math.sigmoid,
    'nn.softmax'              : hlib.nn.softmax,
    'exp'                     : hlib.math.exp,
    'log'                     : hlib.math.log,
    'sqrt'                    : hlib.math.sqrt,
     'nn.conv2d'              : hlib.nn.conv2d,
     'nn.max_pool2d'          : hlib.nn.max_pool2d,
     'transpose'              : hlib.nn.transpose,
     'nn.batch_flatten'       : hlib.nn.flatten,
}

_attrib = {
    'nn.conv2d'               : ['strides','padding','dilation','groups','channels','kernel_size','data_layout','kernel_layout','out_layout','out_dtype'],
    'nn.conv2d_transpose'     : ['channels','kernel_size','strides','padding','output_padding','dilation','groups','data_layout','kernel_layout','out_layout','out_dtype'],
    'nn.max_pool2d'           : ['pool_size','strides','padding','layout'],
    'nn.avg_pool2d'           : ['pool_size','strides','padding','layout'],
    'transpose'               : ['axes'],
    'reshape'                 : ['newshape','reverse'],
    'squeeze'                 : ['axis'],
    'nn.dense'                : ['units','out_dtype'],
    'nn.softmax'              : ['axis'],
    'nn.bias_add'             : ['axis'],
    'tanh'                    : [],
    'nn.relu'                 : [],
    'nn.batch_flatten'        : [],
}

def get_mod(model,shape):
    relay_model=keras.models.load_model(model)
    module,params=frontend.from_keras(relay_model,shape)
    return module,params

def flatten(l):
    _list = []
    for sublist in l:
        if isinstance(sublist,list):
            for item in sublist:
                _list.append(item)
        else:
            _list.append(sublist)
    return _list

def fst(l):
    if isinstance(l,list):
        return fst(l[0])
    else:
        return l

def gen_params(var_list,env):
    args=[]
    params=[]
    for var in var_list:
        if not "%" in var:
            args.append(var)
            params.append(env[var])
    return args,params

def gen_func(params,env,size):
    args = []
    for var in params:
        args.append(var)
    def func(*args):
        for i in range(size):
            key = "%" + str(i)
            name = env[key][0];
            _func = env[key][1];
            _args = env[key][2];
            _kwargs = env[key][3];
            arg_list = []
            print(env[key])
            for var in _args:
                if var in params:
                    arg_list.append(var)
                else:
                    arg_list.append(env[var])
            env[key] = _func(*arg_list,**_kwargs)
            print(env[key].name,":",env[key].shape)
        return env["%" + str(size - 1)]
    return func

def model_extent(func):
    length = 0
    if isinstance(func,Call):
        length = 1
        for arg in func.args:
            if(isinstance(arg,Call)):
                length += model_extent(arg)
        return length
    elif isinstance(func,Let):
        length += model_extent(func.value)
        length += model_extent(func.body)
        return length
    elif isinstance(func,Function):
        length += model_extent(func.body)
        return length
    else:
        return 0

def gen_schedule(args,func):
    return hcl.create_schedule(args,func)

#creating relay_to_hcl parser
def relay_parser(model,shape,frontend='keras',dtype=hcl.Float()):
    hcl.init(dtype)
    def _model_extent(func):
        length = 0
        if hasattr(func,'args'):
            length = 1
            for arg in func.args:
                if(isinstance(arg,Call)):
                    length += _model_extent(arg)
            return length
        else:
            return 0
    if frontend=='keras':
        relay_model=keras.models.load_model(model)
        print(relay_model)
        module,params=relay_front.from_keras(relay_model,shape)
        body = module.functions[module.global_var_map_["main"]]
        place_num = model_extent(body.body)
    elif frontend=='relay':
        body = model
        place_num = model_extent(body)
    def getType(ty):
        pass
    def parse_rec(node,place):
        if isinstance(node,Function):
            print("Function: ")
            var,type_dict,env= parse_rec(node.body,place-1)
        elif isinstance(node,Var):
            name= node.name_hint
            var = [name]
            type_dict = {name:Var}
            env = {}
            if node.name_hint in shape:
                dtype = node.type_annotation.dtype
                env[name]=hcl.placeholder(shape[name],name,dtype)
            else:
                ty = node.type_annotation
                dtype = ty.dtype
                size=[]
                for i in ty.shape:
                    size.append(i.value)
                env[name]=hcl.placeholder(tuple(size),name,dtype)
            print("Var: " + name)
        elif isinstance(node,TupleGetItem):
            index = node.index
            tup   = node.tuple_value
            name  = tup.vid.name_hint
            ty    = tup.type_annotation
            var = [name]
            type_dict = {name:Var}
            env = {}
            tensor = ty.fields[index]
            size = []
            dtype = tensor.dtype
            for i in tensor.shape:
                size.append(i.value)
            env[name]=hcl.placeholder(tuple(size),name,dtype)
            print("TupleGet: " + name)
        elif isinstance(node,Let):
            name = node.var.vid.name_hint
            print("Let: " + name)
            var = [name]
            type_dict = {name:Let}
            env={}
            args=[]
            kwargs={}
            ty = node.var.type_annotation
            dtype = ty.dtype
            size=[]
            arg_len = 0
            temp_len = 0
            for i in ty.shape:
                size.append(i.value)
            bind_var=hcl.placeholder(tuple(size),name,dtype)
            value = node.value
            val_len = model_extent(value)
            temp_var,temp_type,temp_env=parse_rec(value,place-1)
            if isinstance(value,Var):
                env[name] = (name,bind_var,temp_env[fst(temp_var[0])])
            elif isinstance(value,Call):
                var.append(temp_var)
                args.append(temp_env[temp_var[-1]][0])
                temp_len += len(temp_env)
                env.update(temp_env) 
                arg_len = len(var) - temp_len
                for i in range(len(args)):
                    if hasattr(args[i],"name"):
                        if(args[i].name in var):
                            env[args[i].name]=args[i]
                for attr in _attrib[value.op.name]:
                    kwargs[attr] = getattr(value.attrs,attr)
                env[name] = (name,bind_var,_convert_map[value.op.name],tuple(args),kwargs)
            temp_var,temp_type,temp_env=parse_rec(node.body,place-(val_len))
            var.update(temp_var)
            type_dict.update(temp_type)
            env.update(temp_env)
        elif isinstance(node,If):
            pass
        elif isinstance(node,Call):
            print("Call: " + node.op.name)
            name= '%' + str(place)
            args= []
            var = []
            type_dict = {name : Call}
            env = {}
            arg_len = 0
            temp_len = 0
            for arg in node.args:
                temp_var,temp_type,temp_env=parse_rec(arg,place-1)
                if isinstance(arg,Var):
                    var.append(temp_var[0])
                    var = flatten(var)
                    args.append(temp_env[fst(temp_var[0])])
                elif isinstance(arg,Call):
                    var.append(temp_var)
                    var = flatten(var)
                    args.append(temp_env[temp_var[-1]][0])
                    temp_len += len(temp_env)
                    env.update(temp_env)
            type_dict.update(temp_type)
            arg_len = len(var) - temp_len
            var.append(name)
            kwargs={}
            for i in range(len(args)):
                if hasattr(args[i],"name"):
                    if(args[i].name in var):
                        env[args[i].name]=args[i]
            for attr in _attrib[node.op.name]:
                kwargs[attr]=getattr(node.attrs,attr)
            env[name]=(name,_convert_map[node.op.name],tuple(args),kwargs)
        return var,type_dict,env
    out_var,out_type,out_env=parse_rec(body,place_num)
    return out_var,out_type,out_env,place_num,params

def get_relay_model(model,shape={},frontend = 'keras',dtype=hcl.Float()):
    out_var,out_type,out_env,place_num,params=relay_parser(model,shape,frontend)
    arg,_param=gen_params(out_var,out_env)
    for i in _param:
        print(type(i))
    func=gen_func(_param,out_env,place_num)
    print(type(func))
    _inputs = []
    for var in params:
        _inputs.append(hcl.asarray(params[var].asnumpy()))
    s=gen_schedule(_param,func)
    return hcl.build(s),_inputs
