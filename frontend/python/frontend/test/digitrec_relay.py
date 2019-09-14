from relay_parser import *
hcl.init()
_, params = parser_test(d_model, d_params, (10, 784))
_input = hcl.placeholder((10, 784))
bias = hcl.placeholder((784,))
_in = hcl.asarray(np.zeros((10, 784)))
_bias = hcl.asarray(np.zeros(784,))
_out = hcl.asarray(np.zeros((10, 10)))
_params = []
for i in range(len(params)):
    _params.append(hcl.asarray(params[i]))
args = []
inc = 1
for arg in d_params:
    args.append(hcl.placeholder(d_params[arg].shape, "param_" + str(inc)))
    inc = inc + 1


def test(_in, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7):
    dense0 = hlib.nn.dense(_in, arg0)
    b_add0 = hlib.nn.bias_add(dense0, arg1)
    tanh0 = hlib.nn.tanh(b_add0)
    dense1 = hlib.nn.dense(tanh0, arg2)
    b_add1 = hlib.nn.bias_add(dense1, arg3)
    relu0 = hlib.nn.relu(b_add1)
    dense2 = hlib.nn.dense(relu0, arg4)
    b_add2 = hlib.nn.bias_add(dense2, arg5)
    relu1 = hlib.nn.relu(b_add2)
    dense3 = hlib.nn.dense(relu1, arg6)
    b_add3 = hlib.nn.bias_add(dense3, arg7)
    softmax0 = hlib.nn.softmax(b_add3)
    print(softmax0.shape)
    return softmax0


s = hcl.create_schedule([_input, *args], test)
f = hcl.build(s)
f(_in, *_params, _out)
print(_out)
