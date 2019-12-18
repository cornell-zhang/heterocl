import frontend
import tvm.relay.testing as tst
import frontend.relay_parser as rp
net,params = tst.squeezenet.get_workload()
net = net.functions[net.global_var_map_["main"]]
print(net)
print("# of nodes:",rp.model_extent(net))
#v, t, e, pl_n, par = rp.relay_parser(
#    net, (1, 1), frontend='relay')
#print(rp.full_flatten(v))
#print(t)
#print(e)
# frontend.relay_parser.get_relay_model(net,(1,1),frontend='relay',in_params=params)
