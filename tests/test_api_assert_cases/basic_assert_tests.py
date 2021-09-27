import heterocl as hcl
import numpy as np

m = 64
n = 64
k = 64

def test_with_if():
    hcl.init(raise_assert_exception=False)
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))

    def kernel(matrix_1, matrix_2):
        return_matrix = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y], "return_matrix")

        with hcl.if_(matrix_2[0,0] == 0):
            hcl.assert_(matrix_2[1,1] == 0, "assert message in if statement") #result is true
            hcl.print(0, "in the if statement\n") #should be printed

        hcl.assert_(matrix_1[0,0] != 0, "customized assert message 1") #result is false
        hcl.print(0, "this shouldn't be printed")
        return return_matrix

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    return s

def test_with_for():
    hcl.init(raise_assert_exception=False)
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))

    def kernel(matrix_1, matrix_2):
        return_matrix = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y], "return_matrix")

        with hcl.for_(0, 7, name="for_loop") as f:
            hcl.assert_(matrix_2[f,2] == 0, "assert message in the first for loop") #assert true
            hcl.print(0, "in the first for loop\n") #should be printed

        with hcl.for_(0, 7, name="for_loop") as f:
            hcl.assert_(matrix_2[f,2] != 0, "assert message in the second for loop") #assert false
            hcl.print(0, "in the second for loop\n") #should not be printed

        hcl.print(0, "this should not be printed\n") #should not be printed
        return return_matrix

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    return s


def test_for_if():
    hcl.init(raise_assert_exception=False)
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))

    def kernel(matrix_1, matrix_2):
        return_matrix = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y], "return_matrix")
        matrix_A = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 7, "matrix_A")
        matrix_B = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 8, "matrix_B")

        with hcl.for_(0, 7, name="for_loop") as f:
          with hcl.if_(matrix_1[0, f] == 0):
              hcl.assert_(matrix_2[f,2] == 0, "assert message in the first for loop") #assert true
              hcl.print(0, "in the first for loop and if statement\n") #should be printed 7 times

          hcl.print(0, "in the first for loop, outside if statement\n") #should be printed 7 times

        with hcl.for_(0, 7, name="for_loop") as f:
            with hcl.if_(matrix_1[0, f] == 0):
                hcl.assert_(matrix_2[f,2] != 0, "assert message in the second for loop") #assert false
                hcl.print(0, "in the second for loop and if statement\n") #should not be printed

            hcl.print(0, "in the second for loop, outside if statement\n") #should not be printed

        hcl.print(0, "this should not be printed\n") #should not be printed
        matrix_C = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 9, "matrix_C")
        matrix_D = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 10, "matrix_D")
        return return_matrix

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    return s

def test_mem_alloc():
    hcl.init(raise_assert_exception=False)
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))

    def kernel(matrix_1, matrix_2):
        first_matrix = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y], "first_matrix")
        return_matrix = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 7, "return_matrix")

        hcl.assert_(matrix_1[0,0] == 0, "assert %d message % d", [matrix_1[0,0], matrix_2[0,0]]) #assert is true
        hcl.assert_(matrix_1[0,0] == 10, "assert %d message % d number 2", [matrix_1[0,0], matrix_2[0,0]]) #assert is false

        matrix_C = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 9, "matrix_C")
        matrix_D = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 10, "matrix_D")

        hcl.assert_(matrix_1[0,0] == 0, "assert %d message % d number 3", [matrix_1[0,0], matrix_2[0,0]]) #assert is true
        hcl.print(0, "this should not be printed\n") #should not be printed
        return return_matrix

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    return s

def run_tests():
    hcl_m1_if = hcl.asarray(np.zeros((m, n)))
    hcl_m2_if = hcl.asarray(np.zeros((m, n)))
    hcl_m3_if = hcl.asarray(np.zeros((m,n)))

    hcl_m1_for = hcl.asarray(np.zeros((m, n)))
    hcl_m2_for = hcl.asarray(np.zeros((m, n)))
    hcl_m3_for = hcl.asarray(np.zeros((m,n)))

    hcl_m1_for_if = hcl.asarray(np.zeros((m, n)))
    hcl_m2_for_if = hcl.asarray(np.zeros((m, n)))
    hcl_m3_for_if = hcl.asarray(np.zeros((m,n)))

    hcl_m1_mem = hcl.asarray(np.zeros((m, n)))
    hcl_m2_mem = hcl.asarray(np.zeros((m, n)))
    hcl_m3_mem = hcl.asarray(np.zeros((m,n)))

    s_if = test_with_if()
    s_for = test_with_for()
    s_for_if = test_for_if()
    s_mem = test_mem_alloc()

    #test assert in if statment
    f_if = hcl.build(s_if)
    f_if(hcl_m1_if, hcl_m2_if, hcl_m3_if)

    #test assert in for loop
    f_for = hcl.build(s_for)
    f_for(hcl_m1_for, hcl_m2_for, hcl_m3_for)

    #test assert in if statment in a for loop
    f_for_if = hcl.build(s_for_if)
    f_for_if(hcl_m1_for_if, hcl_m2_for_if, hcl_m3_for_if)

    #test assert free memory
    f_mem = hcl.build(s_mem)
    f_mem(hcl_m1_mem, hcl_m2_mem, hcl_m3_mem)

run_tests()
