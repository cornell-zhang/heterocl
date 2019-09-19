import frontend
import tvm.relay.testing as tst
import frontend.relay_parser as rp
import heterocl as hcl
#net = tst.lstm.lstm_cell(1)
hcl.init(hcl.Float())
net,params = tst.lstm.get_workload(10, 1)
net = net.functions[net.global_var_map_["main"]]
print(net)
#print("# of nodes:",rp.model_extent(net))
#v, t, e, pl_n, par = rp.relay_parser(
#    net, (1, 1), frontend='relay')
#print("Var:",rp.full_flatten(v))
#print("Dict:",t)
#env = []
#dic = []
#for en in e:
#    env.append(en)
#    dic.append(e[en])
#print("Env_keys:",env)
#print("Env:",dic)
frontend.relay_parser.get_relay_model(net,(1,1),frontend='relay',in_params=params)
