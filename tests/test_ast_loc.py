import heterocl as hcl
import numpy as np
import pytest

@pytest.mark.skip(reason="data type inference to be supported")
def test_unary():
    shape = (1,10)

    def loop_body(data, power):
        D = hcl.power(5,2)
        E = hcl.compute(shape, lambda x,y: hcl.power(data[x,y],power[x,y]), "loop_body")
        print(E.loc)
        print(D.loc)
        return E

    A = hcl.placeholder(shape, "A")
    B = hcl.placeholder(shape, "B")
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

    print(A.loc)
    print(B.loc)
    
    assert np.allclose(1, 2)
