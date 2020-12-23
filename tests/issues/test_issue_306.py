import heterocl as hcl

def test_inter_kernel_channels():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    C = hcl.placeholder((10, 32), "C")
    def kernel(A, C):
        
        B = hcl.compute((10, 32), lambda *args: 0, "B")
        @hcl.def_([(10, 32), (10, 32)])
        def add(A, B):
            hcl.update(B, lambda *args: A[args] + 1)

        @hcl.def_([(10, 32), (10, 32)])
        def mul(B, C):
            hcl.update(C, lambda *args: B[args] * 2)
            
        add(A, B)
        mul(B, C)
    
    s = hcl.create_schedule([A, C], kernel)

    s.to(kernel.mul.B, kernel.add.B, depth=10)
    # s.to(kernel.add.B, kernel.mul.B, depth=10)
    code = str(hcl.lower(s))
    print(code)


if __name__ == '__main__':
    test_inter_kernel_channels()
