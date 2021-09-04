import heterocl as hcl
import numpy as np

m = 64
n = 64
k = 64

def test_mem_alloc_nested():
    hcl.init(raise_assert_exception=False)
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))

    def kernel(matrix_1, matrix_2):
        first_matrix = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y], "first_matrix")
        return_matrix = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y] + 7, "return_matrix")
        ax = hcl.scalar(0)
        with hcl.while_(ax.v < 3):
            matrix_A = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y] + 7, "matrix_A")

            with hcl.for_(0, 2, name="for_loop_in") as h:
                matrix_B = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y] + 8, "matrix_B")

                with hcl.if_(matrix_1[0, 2] >= 0):
                    matrix_C = hcl.compute((m, k), lambda x, y : matrix_1[x, x] + matrix_2[x, x] + 9, "matrix_C")
                    hcl.assert_(matrix_1[0, 0]> 0, "assert message in the if statement %d", matrix_C[0, 0])
                    matrix_D = hcl.compute((m, k), lambda x, y : matrix_1[x, x] + matrix_2[x, x] + 9, "matrix_D")
                    hcl.print(0, "in if statement\n")

                hcl.assert_(matrix_1[0, 0]> 1, "assert message for loop")
                matrix_F = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y] + 8, "matrix_F")
                hcl.print(0, "in for loop\n")

            hcl.assert_(matrix_1[0, 0]> 2, "assert error, matrix_A[1, 1]: %d matrix_A[2, 1]: %d matrix_A[3, 1]: %d", [matrix_A[1, 1], matrix_A[2, 1], matrix_A[3, 1]])
            hcl.print(0, "in the while loop\n")
            ax.v = ax.v + 1

        hcl.assert_(matrix_1[0, 0]> 3, "assert message end")
        matrix_E = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y] + 10, "matrix_E")
        hcl.print(0, "this should not be printed\n")
        return return_matrix

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    return s

def test_mem_if():
    hcl.init(raise_assert_exception=False)
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))
    def kernel(matrix_1, matrix_2):
        return_matrix = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y], "return_matrix")

        with hcl.if_(matrix_2[0, 0] == 0):
            matrix_A = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y], "matrix_A")
        with hcl.else_():
            matrix_B = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y] + 2, "matrix_B")

        hcl.assert_(matrix_1[0, 0] != 0, "customized assert message 1") #result is false

        matrix_C = hcl.compute((m, k), lambda x, y : matrix_1[x, y] + matrix_2[x, y], "matrix_C")
        hcl.print(0, "this shouldn't be printed")
        return return_matrix

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    return s

def test_mutate():
    hcl.init(raise_assert_exception=False)
    A = hcl.placeholder((10, ))
    M = hcl.placeholder((2, ))
    def kernel(A, M):
        def loop_body(x):
            with hcl.if_(A[x]> M[0]):
                with hcl.if_(A[x]> M[1]):
                    hcl.assert_(x == 2, "assert error in if--value of x: %d", x)
                    M[0] = M[1]
                    M[1] = A[x]
                with hcl.else_():
                    M[0] = A[x]
        hcl.mutate(A.shape, lambda x : loop_body(x))
        hcl.print(0, "this should not be printed\n")
    s = hcl.create_schedule([A, M], kernel)
    return s

def run_tests():
    hcl_m1_mem0 = hcl.asarray(np.zeros((m, n)))
    hcl_m2_mem0 = hcl.asarray(np.zeros((m, n)))
    hcl_m3_mem0 = hcl.asarray(np.zeros((m,n)))

    hcl_m1_mem1 = hcl.asarray(np.ones((m, n)))
    hcl_m2_mem1 = hcl.asarray(np.ones((m, n)))
    hcl_m3_mem1 = hcl.asarray(np.zeros((m,n)))

    np_mem2 = 2 * np.ones((m,n))
    hcl_m1_mem2 = hcl.asarray(np_mem2)
    hcl_m2_mem2 = hcl.asarray(np_mem2)
    hcl_m3_mem2 = hcl.asarray(np.zeros((m,n)))

    np_mem3 = 3 * np.ones((m,n))
    hcl_m1_mem3 = hcl.asarray(np_mem3)
    hcl_m2_mem3 = hcl.asarray(np_mem3)
    hcl_m3_mem3 = hcl.asarray(np.zeros((m,n)))

    s_mem = test_mem_alloc_nested()

    f_mem = hcl.build(s_mem)

    #these tests check whether assert false deallocates memory properly when in nested loops
    f_mem(hcl_m1_mem0, hcl_m2_mem0, hcl_m3_mem0)

    f_mem(hcl_m1_mem1, hcl_m2_mem1, hcl_m3_mem1)

    f_mem(hcl_m1_mem2, hcl_m2_mem2, hcl_m3_mem2)

    f_mem(hcl_m1_mem3, hcl_m2_mem3, hcl_m3_mem3)

    hcl_m1_if = hcl.asarray(np.zeros((m, n)))
    hcl_m2_if = hcl.asarray(np.zeros((m, n)))
    hcl_m3_if = hcl.asarray(np.zeros((m,n)))

    s_if = test_mem_if()

    #this is to test the case where memory gets allocated and then deallocated before assert statement
    f_if = hcl.build(s_if)
    f_if(hcl_m1_if, hcl_m2_if, hcl_m3_if)

    hcl_m1_mutate = hcl.asarray(np.ones((10,)))
    hcl_m2_mutate = hcl.asarray(np.zeros((2,)))
    hcl_m3_mutate = hcl.asarray(np.zeros((m,n)))

    s_mutate = test_mutate()

    #this is to test whether assert is compatible with mutate
    f_mutate = hcl.build(s_mutate)
    f_mutate(hcl_m1_mutate, hcl_m2_mutate)

run_tests()
