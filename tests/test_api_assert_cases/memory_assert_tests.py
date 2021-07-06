import heterocl as hcl
import numpy as np
    
m = 64
n = 64
k = 64

def test_mem_alloc_nested():

    Dtype = hcl.Int()
    hcl.init(Dtype)
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))

    def kernel(matrix_1, matrix_2):
        first_matrix = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y], dtype=Dtype, name="first_matrix")
        return_matrix = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 7, dtype=Dtype, name="return_matrix")
        
        ax = hcl.scalar(0)
        with hcl.while_(ax.v < 7):
          matrix_A = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 7, dtype=Dtype, name="matrix_A")
          
          with hcl.for_(0, 7, name="for_loop_in") as h:
            matrix_B = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 8, dtype=Dtype, name="matrix_B")
            
            with hcl.if_(matrix_1[0, 2] >= 0):
              matrix_C = hcl.compute((m,k), lambda x, y: matrix_1[x, x] + matrix_2[x, x] + 9, dtype=Dtype, name="matrix_C")
              hcl.assert_(matrix_1[0, 0] > 0, message="assert message in the if statement") 
              matrix_D = hcl.compute((m,k), lambda x, y: matrix_1[x, x] + matrix_2[x, x] + 9, dtype=Dtype, name="matrix_D")
              hcl.print(0, "in if statement \n")
            
            hcl.assert_(matrix_1[0, 0] > 1, message="assert message for loop")
            matrix_F = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 8, dtype=Dtype, name="matrix_F")
            hcl.print(0, "in for loop \n")
            
          hcl.assert_(matrix_1[0, 0] > 2, message="assert message while loop") 
          hcl.print(0, format="in the while loop\n")
          ax.v = ax.v + 1
                       
        hcl.assert_(matrix_1[0, 0] > 3, message="assert message end")
        matrix_E = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 10, dtype=Dtype, name="matrix_E")
        hcl.print(0, "this should not be printed\n") 
        
        return return_matrix
        
    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    
    return s
    
def test_mem_if():

    Dtype = hcl.Int()
    hcl.init(Dtype)
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))

    def kernel(matrix_1, matrix_2):
        return_matrix = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y], dtype=Dtype, name="return_matrix")
   
        with hcl.if_(matrix_2[0,0] == 0):
            matrix_A = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y], dtype=Dtype, name="matrix_A")
        with hcl.else_():
            matrix_B = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y] + 2, dtype=Dtype, name="matrix_B")
            
        hcl.assert_(matrix_1[0,0] != 0, message="customized assert message 1") #result is false
        
        matrix_C = hcl.compute((m,k), lambda x, y: matrix_1[x,y] + matrix_2[x,y], dtype=Dtype, name="matrix_C")
   
        hcl.print(0, "this shouldn't be printed")

        return return_matrix
        
    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    
    return s
    
   
def run_tests():
    Dtype = hcl.Int()
    
    hcl_m1_mem0 = hcl.asarray(np.zeros((m, n)), Dtype)
    hcl_m2_mem0 = hcl.asarray(np.zeros((m, n)), Dtype)
    hcl_m3_mem0 = hcl.asarray(np.zeros((m,n)), Dtype)
    
    hcl_m1_mem1 = hcl.asarray(np.ones((m, n)), Dtype)
    hcl_m2_mem1 = hcl.asarray(np.ones((m, n)), Dtype)
    hcl_m3_mem1 = hcl.asarray(np.zeros((m,n)), Dtype)
    
    np_mem2 = 2 * np.ones((m,n))
    print(np_mem2)
    hcl_m1_mem2 = hcl.asarray(np_mem2, Dtype)
    hcl_m2_mem2 = hcl.asarray(np_mem2, Dtype)
    hcl_m3_mem2 = hcl.asarray(np.zeros((m,n)), Dtype)
    
    np_mem3 = 3 * np.ones((m,n))
    print(np_mem3)
    hcl_m1_mem3 = hcl.asarray(np_mem3, Dtype)
    hcl_m2_mem3 = hcl.asarray(np_mem3, Dtype)
    hcl_m3_mem3 = hcl.asarray(np.zeros((m,n)), Dtype)
    
    s_mem = test_mem_alloc_nested()
    
    f_mem = hcl.build(s_mem)
     
    #these tests check whether assert false deallocates memory properly when in nested loops 
    print(" \n")
    print("test assert")
    print("----------------------------------")
    
    f_mem(hcl_m1_mem0, hcl_m2_mem0, hcl_m3_mem0) 
    
    print(" \n")
    print("test assert")
    print("----------------------------------")
    
    f_mem(hcl_m1_mem1, hcl_m2_mem1, hcl_m3_mem1)
     
    print(" \n")
    print("test assert")
    print("----------------------------------")
    
    f_mem(hcl_m1_mem2, hcl_m2_mem2, hcl_m3_mem2)
     
    print(" \n")
    print("test assert")
    print("----------------------------------")
    
    f_mem(hcl_m1_mem3, hcl_m2_mem3, hcl_m3_mem3)
    
    
    hcl_m1_if = hcl.asarray(np.zeros((m, n)), Dtype)
    hcl_m2_if = hcl.asarray(np.zeros((m, n)), Dtype)
    hcl_m3_if = hcl.asarray(np.zeros((m,n)), Dtype)
    
    s_if = test_mem_if()
    
    #this is to test the case where memory gets allocated and then deallocated before assert statement
    print("test mem if")
    print("----------------------------------")
    
    f_if = hcl.build(s_if)
    f_if(hcl_m1_if, hcl_m2_if, hcl_m3_if)  
    
run_tests()