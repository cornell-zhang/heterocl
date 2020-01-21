import heterocl as hcl
import numpy as np
import hlib
import numpy.testing as tst

def _test_transpose(shape, axes):
    hcl.init()
    I = hcl.placeholder(shape)

    def transpose(I, axes=axes):
        return hlib.op.nn.transpose(I, axes)
    s = hcl.create_schedule([I], transpose)
    f = hcl.build(s)
    data = np.random.randint(50, size=shape)
    _out = hcl.asarray(np.transpose(np.zeros(shape), axes=axes))
    data = hcl.asarray(data)
    f(data, _out)
    t_out = np.transpose(data.asnumpy(), axes=axes)
    tst.assert_almost_equal(t_out, _out.asnumpy())


def test_transpose():
    _test_transpose((4, 3, 2, 1), [0, 2, 3, 1])
    _test_transpose((3, 3, 3), [0, 2, 1])
    _test_transpose((2, 2, 2, 2, 2, 2), [0, 2, 3, 1, 5, 4])
