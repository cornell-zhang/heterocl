from wsgiref.handlers import IISCGIHandler
import heterocl as hcl
import hcl_mlir
import numpy as np
import onnx

from partition import *
import sys

##Generate dfg from onnx file. Leverage mainly the graph topology


def parse_tensor(tensor_info):
    return onnx.helper.printable_value_info(tensor_info).split("[")[1].split(']')[0].split(', ')

def build_table(model):
    tab = {}
    for e in model.graph.value_info:
        tab[e.name] = parse_tensor(e)
    for e in model.graph.input:
        tab[e.name] = parse_tensor(e)
    for e in model.graph.output:
        tab[e.name] = parse_tensor(e)

    return tab

def onnx_to_dfg(model):
    tab = build_table(model)
    input_info = parse_tensor(model.graph.input[0])
    dtype = hcl.Int()
    if(input_info[0] == "FLOAT"):
        dtype = hcl.Float()
    print(input_info)
    #TODO : Make more general
    input_shape = None
    if "i" in str(input_info[1]) :
        input_shape = [0]
    else :
        input_shape = list(map(int,input_info[1].split('x')))
    #print(list(map(int,input_info[1].split('x'))))
    input = hcl.placeholder(input_shape, model.graph.node[0].input[0], dtype )
    graph = hcl.DataflowGraph(model.graph.name,[input])
    n = len(model.graph.node)
    for i in range(n):
        name_i = model.graph.node[i].output[0]
        info_i = ["","0"]
        if name_i in tab.keys() : #handle the case where value_info is empty
            info_i = tab[name_i]
        dtype_i = hcl.Int()
        if(info_i[0] == "FLOAT"):
            dtype_i = hcl.Float()

        infoi_shape = None
        if "i" in str(info_i[1]) :
            infoi_shape = [0]
        else  :
            infoi_shape = list(map(int,info_i[1].split('x')))
        
        ip = graph.create_node(hcl.placeholder(infoi_shape, model.graph.node[i].op_type + "_" +name_i , dtype_i ))
        if input.name in model.graph.node[i].input:
            graph.add_edge(input, ip)
        for j in range(i+1,n):
            if model.graph.node[i].output[0] in model.graph.node[j].input:
                name_o = model.graph.node[j].output[0]
                info_o = ["","0"]
                if name_o in tab.keys() : #handle the case where value_info is empty
                    info_o = tab[name_o]
                dtype_o = hcl.Int()
                if(info_o[0] == "FLOAT"):
                    dtype_o = hcl.Float()

                infoo_shape = None
                if "i" in str(info_o[1]) :
                    infoo_shape = [0]
                else  :
                    infoo_shape = list(map(int,info_o[1].split('x')))

                ot = graph.create_node(hcl.placeholder(infoo_shape, model.graph.node[j].op_type + "_" +name_o , dtype_o ))
                graph.add_edge(ip, ot)
    #print(graph.node_map["conv1/7x7_s2_1"].children[0].name)
    return graph

gnet = onnx_to_dfg(onnx.load('/home/yassine/Downloads/googlenet-12.onnx'))
gnet.visualize()
print(gnet)
# with open('gpt2_10-results-EA','w') as f:
#     for c in range(100):
#         print("step",c)
#         f.write(str(partition_graph(gnet)))