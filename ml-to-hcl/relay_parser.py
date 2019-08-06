import tvm,keras
import tvm.relay.frontend as frontend
import numpy as np
import heterocl as hcl
import hlib
from tvm.relay.expr import Function,Var,Call
from tvm.relay.ty import TensorType

#Exploring the AST here

_convert_map = {
    'nn.dense'                : hlib.nn.dense,
    'nn.relu'                 : hlib.nn.relu,
    'nn.bias_add'             : hlib.nn.bias_add,
#    'prelu'                   : 'hlib.nn.prelu',
    'tanh'                    : hlib.nn.tanh,
#    'sigmoid'                 : 'hlib.nn.sigmoid',
    'nn.softmax'                 : hlib.nn.softmax,
#    'exp'                     : 'hlib.exp',
#    'log'                     : 'hlib.log',
#    'sqrt'                    : 'hlib.sqrt',
     'nn.conv2d'              : hlib.nn.conv2d,
     'nn.max_pool2d'          : hlib.nn.max_pool2d,
     'transpose'              : hlib.nn.transpose,
     'nn.batch_flatten'       : hlib.nn.flatten,
}

_attrib = {
    'conv2d'                  : ['strides','padding','dilation','layout'],
    'max_pool2d'              : ['pool_size','strides','padding'],
    'transpose'               : ['axes'],
    'dense'                   : [],
    'softmax'                 : [],
    'flatten'                 : [],
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
            arg_list = []
            for var in _args:
                if var in params:
                    arg_list.append(var)
                else:
                    arg_list.append(env[var])
            env[key] = _func(*arg_list)
            print(env[key].name,":",env[key].shape)
        return env["%" + str(size - 1)]
    return func

def gen_schedule(args,func):
    return hcl.create_schedule(args,func)

#creating relay_to_hcl parser
def relay_parser(model,shape,dtype=hcl.Float()):
    hcl.init(dtype)
    relay_model=keras.models.load_model(model)
    module,params=frontend.from_keras(relay_model,shape)
    def model_extent(func):
        length = 0
        if hasattr(func,'args'):
            length = 1
            for arg in func.args:
                if(isinstance(arg,Call)):
                    length += model_extent(arg)
            return length
        else:
            return 0
    place_num = model_extent(module.body)
    def parse_rec(node,place):
        if isinstance(node,Function):
            print("Function: ")
            var,env= parse_rec(node.body,place-1)
        elif isinstance(node,Var):
            name= node.name_hint
            var = [name]
            env = {}
            if node.name_hint.find('input')>=0:
                dtype = 'float32'
                env[name]=hcl.placeholder(shape,name,dtype)
            else:
                ty = node.type_annotation
                size=[]
                for i in ty.shape:
                    size.append(i.value)
                env[name]=hcl.placeholder(tuple(size),name)
            print("Var: " + name)
        elif isinstance(node,Call):
            print("Call: " + node.op.name)
            name= '%' + str(place)
            args= []
            var = []
            env = {}
            arg_len = 0
            temp_len = 0
            for arg in node.args:
                temp_var,temp_env=parse_rec(arg,place-1)
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
            arg_len = len(var) - temp_len
            var.append(name)
            for i in range(len(args)):
                if hasattr(args[i],"name"):
                    if(args[i].name in var):
                        env[args[i].name]=args[i]
                if hasattr(args[i],"attrs"):
                    for attr in _attrib[node.op.name]:
                        pass
            env[name]=(name,_convert_map[node.op.name],tuple(args))
        return var,env
    out_var,out_env=parse_rec(module,place_num)
    return out_var,out_env,place_num,params

def get_relay_model(model,shape=0,dtype=hcl.Float()):
    out_var,out_env,place_num,params=relay_parser(model,shape)
    arg,_param=gen_params(out_var,out_env)
    func=gen_func(_param,out_env,place_num)
    _inputs = []
    for var in params:
        _inputs.append(hcl.asarray(params[var].asnumpy()))
    s=gen_schedule(_param,func)
    return hcl.build(s),_inputs
