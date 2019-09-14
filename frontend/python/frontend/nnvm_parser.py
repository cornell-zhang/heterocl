# -------------------------------------------------------------------------
# NNVM wrapper to convert ML into ir and parameter arrays that are
# saved into json and pickle files for HeteroCL to use on a seperate
# environment
# -------------------------------------------------------------------------
import nnvm
import tvm
import keras
import nnvm.graph as _graph
import numpy as np
import heterocl as hcl
import hlib
import json
import os
import pickle
import inspect
from ast import literal_eval
_convert_map = {
    'dense': hlib.nn.dense,
    'relu': hlib.nn.relu,
    #    'prelu'                   : 'hlib.nn.prelu',
    'tanh': hlib.math.tanh,
    #    'sigmoid'                 : 'hlib.nn.sigmoid',
    'softmax': hlib.nn.softmax,
    #    'exp'                     : 'hlib.exp',
    #    'log'                     : 'hlib.log',
    #    'sqrt'                    : 'hlib.sqrt',
    'conv2d': hlib.nn.conv2d,
    'max_pool2d': hlib.nn.max_pool2d,
    'transpose': hlib.nn.transpose,
    'flatten': hlib.nn.flatten,
}

_attrib = {
    'conv2d': ['strides', 'padding', 'dilation', 'layout'],
    'max_pool2d': ['pool_size', 'strides', 'padding'],
    'transpose': ['axes'],
    'dense': [],
    'softmax': [],
    'flatten': [],
}

_has_layout = ['conv2d', 'max_pool2d']

_convert_str = {
    'dense': 'hlib.nn.dense',
    'relu': 'hlib.nn.relu',
    'tanh': 'hlib.nn.tanh',
    'sigmoid': 'hlib.nn.sigmoid',
    'exp': 'hlib.exp',
    'log': 'hlib.log',
    'sqrt': 'hlib.sqrt',
    'elemwise_add': 'hlib.elemwise_add',
    'elemwise_sub': 'hlib.elemwise_sub',
    'elemwise_mul': 'hlib.elemwise_mul',
    'elemwise_div': 'hlib.elemwise_div',
    'elemwise_sum': 'hlib.elemwise_sum',
    'flatten': 'hlib.nn.flatten',
    'concatenate': 'hlib.nn.concatenate',
    'expand_dims': 'hlib.expand_dims',
    'squeeze': 'hlib.squeeze',
    'split': 'hlib.split',
    'droupout': 'hlib.dropout',
    'batch_norm': 'hlib.batch_norm',
    'softmax': 'hlib.nn.softmax',
    'log_softmax': 'hlib.nn.log_softmax',
    'pad': 'hlib.pad',
    'block_grad': 'hlib.block_grad',
    # convolutions
    'conv2d': 'hlib.nn.conv2d',
    'conv2d_transpose': 'hlib.nn.conv2d_transpose',
    'max_pool2d': 'hlib.nn.max_pool2d',
    'avg_pool2d': 'hlib.nn.avg_pool2d',
    'global_max_pool2d': 'hlib.nn.global_max_pool2d',
    'global_avg_pool2d': 'hlib.nn.global_avg_pool2d',

    'transpose': 'hlib.transpose',
}

#used in function


def gen_placeholders(sym, node, batch_size, shape, is_param):
    size = []
    if(is_param):
        sym[node] = hcl.placeholder(shape, node)
    else:
        size.append(batch_size)
        for i in shape:
            size.append(i)
        sym[node] = hcl.placeholder(tuple(size), node)
#used in function


def gen_function(sym, nodes, head, layout):
    args = []
    for j in sym:
        args.append(sym[j])
    f_ind = []
    for ind in range(len(nodes)):
        if not nodes[ind]['op'] == 'null':
            f_ind.append(ind)

    def func(*args):
        for ind in f_ind:
            arg = []
            for _input in nodes[ind]['inputs']:
                arg.append(sym[nodes[_input[0]]['name']])
            if 'attrs' in nodes[ind]:
                attrs = _attrib[nodes[ind]['op']]
                print(attrs)
                for attr in attrs:
                    if attr in nodes[ind]['attrs']:
                        arg.append(
                            tuple(literal_eval(str(nodes[ind]['attrs'][attr]))))
            if nodes[ind]['op'] in _has_layout:
                arg.append(layout)
            arg = tuple(arg)
            print(arg)
            symbol = _convert_map[nodes[ind]['op']](*arg)
            sym[nodes[ind]['name']] = symbol
        return sym[nodes[head]['name']]
    return func


def gen_schedule(arg, func):
    return hcl.create_schedule(arg, func)
#used in function


def save_json(_json, name):
    with open(name + '_params/' + name + '.json', 'w') as outfile:
        json.dump(_json, outfile)


def save_file(name, code, target):
    if target == 'vhls':
        with open(name + '.c', 'w') as outfile:
            outfile.write(code)
#used in function


def save_obj(obj, name):
    with open(name + '_params/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# converts the nnvm to ir for heterocl to use


def from_nnvm(
        model,
        frontend,
        filename,
        input_shape=None,
        batch_size=None,
        layout="NCHW",
        dtype=hcl.Float(),
        target=None):
    if frontend == "keras":
        nnvm_model = keras.models.load_model(model)
        graph, params = nnvm.frontend.from_keras(nnvm_model)
    else:
        raise NameError('frontend {} is not a valid frontend'.format(frontend))
    # generate json from model
    graph = _graph.create(graph)
    _json = json.loads(graph.json())
    nodes = _json["nodes"]
    param_shape = {}
    # convert params to dict of numpy arrays
    for param in params:
        params[param] = params[param].asnumpy()
        param_shape[param] = params[param].shape
    # add shapes to _json for lower function
    _json["param_shape"] = param_shape
    var_sym = {}
    hcl.init(dtype)
    # generate placeholders
    for node in _json["arg_nodes"]:
        new_node = nodes[node]
        node_name = new_node['name']
        if 'input' in node_name:
            gen_placeholders(
                var_sym,
                node_name,
                batch_size,
                input_shape,
                False)
        else:
            gen_placeholders(
                var_sym,
                node_name,
                batch_size,
                params[node_name].shape,
                True)
    args = []
    for j in var_sym:
        args.append(var_sym[j])
    func = gen_function(var_sym, nodes, _json['heads'][0][0], layout)
    s = gen_schedule(args, func)
    # transform params so they can be used in function
    param = []
    for i in params:
        param.append(hcl.asarray(params[i]))
    if target is None:
        return hcl.build(s), tuple(param)
    else:
        try:
            f = hcl.build(s, target=target, name=filename)
            save_file(filename, f, target)
            return f, tuple(param)
        except ValueError:
            print("target {} provided is not a compatible target".format(target))
