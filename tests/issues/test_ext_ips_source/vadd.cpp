#include <iostream> 
#include <vector> 

void vec_add(int* A, int* B, int* out, int size) {
    for (int k = 0; k < size; k++) {
        out[k] = A[k] + B[k];
    }
}
