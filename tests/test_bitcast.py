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

    _A_np = np.random.randint(100, size=(10,10)).astype(np.uint32)
    _A = hcl.asarray(_A_np, dtype=hcl.UInt(32))
    _B = hcl.asarray(np.zeros((10,10)), dtype=hcl.Float(32))

    f(_A, _B)

    _B = _B.asnumpy()
    answer = np.frombuffer(_A_np.tobytes(), np.float32).reshape((10,10))

    assert np.array_equal(_B, answer)

def test_bitcast_float2uint():
    hcl.init()
    A = hcl.placeholder((10,10), dtype=hcl.Float(32))

    def algorithm(A):
        B = hcl.bitcast(A, hcl.UInt(32))
        return B
        
    s = hcl.create_schedule([A], algorithm)
    f = hcl.build(s)

    print(hcl.lower(s))

    _A_np = np.random.rand(10,10).astype(np.float32)
    _A = hcl.asarray(_A_np, dtype=hcl.Float(32))
    _B = hcl.asarray(np.zeros((10,10)), dtype=hcl.UInt(32))

    f(_A, _B)

    _B = _B.asnumpy()
    answer = np.frombuffer(_A_np.tobytes(), np.uint32).reshape((10,10))
    
    assert np.array_equal(_B, answer); 

if __name__ == "__main__" :
    test_bitcast_uint2float()
    test_bitcast_float2uint()
