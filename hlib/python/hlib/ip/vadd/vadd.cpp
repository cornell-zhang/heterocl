#include <ap_int.h>
#include <ap_fixed.h>

extern "C" {
    void vadd(ap_int<32>* A, ap_int<32>* B, ap_int<32>* ret) {
        for (size_t k = 0; k < length; k++) {
            ret[k] = A[k] + B[k];
        }
    }
}
