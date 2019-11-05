import frontend
import tvm.relay.testing as tst
import frontend.relay_parser as rp

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_test.shape)
x = x_test[0:9984,0:32,0:32,0:3]/255
#x = x_train[0:49984,0:32,0:32,0:3]/255#.reshape(-1, 32,32,32,3)#.transpose(0,1,4,2,3)
x=np.reshape(x,(-1,32,32,32,3))
x=np.transpose(x,[0,1,4,2,3])
x= np.reshape(x,(-1,32,3,32,32))
y = y_test[0:9984].reshape(-1,32)
test = 1
if(test == 0):
    test_wrapper("cifar10.h5", "nnvm", "keras", "cifar10",
                 x, y, (10, 10), (784,), 10)
elif(test == 1):
    test_wrapper("cifar10.h5", "relay", "keras", "cifar10",
                 x, y, (32, 10), {'input_1': (32,3,32,32)}, batch_size=32)
elif(test == 2):
    f, params = from_nnvm("model.hdf5", 'keras', 'digitrec', (784,), 10)
    print(type(f))
    print(type(params[0]))
    for i in range(6000):
        _in = hcl.asarray(x_train[i].reshape(10, 784))
        _out = hcl.asarray(np.zeros((10, 10)))
        f(_in, *params, _out)
        # print(np.argmax(_out.asnumpy(),axis=1),y_train[i])
elif(test == 3):
    f, params = relay_parser.get_relay_model("cifar10.h5", {"input_1":(32, 3,32,32)})
    for i in range(1562):
        _in = np.transpose(x_train[i],[0,3,1,2])
        _in = hcl.asarray(_in.reshape(32,3,32,32))
        _out = hcl.asarray(np.zeros((32, 10)))
        f(_in, *params, _out)
        print(np.argmax(_out.asnumpy(), axis=1), y_train[i])