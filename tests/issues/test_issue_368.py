import heterocl as hcl

def test_tensor_slice_shape():
    A = hcl.compute((2,10), lambda i,j: 0)

    assert A[0].shape == (10,)
