import heterocl as hcl 
import numpy as np


def test_bitcast_uint2float():
    hcl.init()
    A = hcl.placeholder((10,10), dtype=hcl.UInt(32))
    
    def algorithm(A):
        B = hcl.bitcast(A, hcl.Float(32))
        return B
        

    s = hcl.create_schedule([A], algorithm)
    f = hcl.build(s)

    print(hcl.lower(s))

    _A = hcl.asarray(np.random.randint(100, size=(10,10)), dtype=hcl.UInt(32))
    _B = hcl.asarray(np.zeros((10,10)), dtype=hcl.Float(32))

    f(_A, _B)

    _B = _B.asnumpy
    print(_B)

def test_bitcast_float2uint():
    hcl.init()
    A = hcl.placeholder((10,10), dtype=hcl.Float(32))

    def algorithm(A):
        B = hcl.bitcast(A, hcl.UInt(32))
        return B
        
    s = hcl.create_schedule([A], algorithm)
    f = hcl.build(s)

    print(hcl.lower(s))

    _A = hcl.asarray(np.random.randint(100, size=(10,10)), dtype=hcl.Float(32))
    _B = hcl.asarray(np.zeros((10,10)), dtype=hcl.UInt(32))

    f(_A, _B)

    _B = _B.asnumpy
    print(_B)

if __name__ == "__main__" :
    test_bitcast_uint2float()
    test_bitcast_float2uint()