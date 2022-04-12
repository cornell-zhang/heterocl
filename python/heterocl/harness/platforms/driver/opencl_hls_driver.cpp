#pragma message("driver code for Vitis HLS backend")

// Enable test and run "gcc -E opencl_hls_driver.cpp"
#define TEST

// ===== HLS top function information =====
#ifdef TEST
#define FUNCTION_TYPE __kernel
#define FUNCTION_ATTR __attribute((num_compute_units(2))) __attribute((reqd_work_group_size(128, 1, 1)))
#define FUNCTION_NAME top
#define NUM_OF_ARGS 2

#define argType0 int *restrict
#define argName0 A
#define argDim0
#define argParam0

#define argType1 int *
#define argName1 C
#define argDim1
#define argParam1
#endif
// ====== End of configurable knobs =====

// OpenCL include extra options
#define DO_PRAGMA(x) _Pragma(#x)
DO_PRAGMA(OPENCL EXTENSION cl_khr_global_int32_base_atomics
          : enable)
DO_PRAGMA(OPENCL EXTENSION cl_khr_global_int32_extended_atomics
          : enable)

// macro utility functions
#define PORT(argType, argName, argDim) argType argName argDim
#define INTERFACE_PRAGMA(argName, argParam) \
    DO_PRAGMA(HLS interface port = argName argParam)

#define DUP_ARGS(n, c) DUP_ARGS##n(c)
#define DUP_ARGS7(c) c(argType0, argName0, argDim0), c(argType1, argName1, argDim1), c(argType2, argName2, argDim2), \
                     c(argType3, argName3, argDim3), c(argType4, argName4, argDim4), c(argType5, argName5, argDim5), c(argType6, argName6, argDim6)
#define DUP_ARGS6(c) c(argType0, argName0, argDim0), c(argType1, argName1, argDim1), c(argType2, argName2, argDim2), \
                     c(argType3, argName3, argDim3), c(argType4, argName4, argDim4), c(argType5, argName5, argDim5)
#define DUP_ARGS5(c) c(argType0, argName0, argDim0), c(argType1, argName1, argDim1), c(argType2, argName2, argDim2), \
                     c(argType3, argName3, argDim3), c(argType4, argName4, argDim4)
#define DUP_ARGS4(c) c(argType0, argName0, argDim0), c(argType1, argName1, argDim1), c(argType2, argName2, argDim2), \
                     c(argType3, argName3, argDim3)
#define DUP_ARGS3(c) c(argType0, argName0, argDim0), c(argType1, argName1, argDim1), c(argType2, argName2, argDim2)
#define DUP_ARGS2(c) c(argType0, argName0, argDim0), c(argType1, argName1, argDim1)
#define DUP_ARGS1(c) c(argType0, argName0, argDim0)
#define DUP_ARGS0(c)

#define PORTS(num, c) DUP_ARGS(num, c)

#ifdef TEST
// define function name and ports
FUNCTION_TYPE
FUNCTION_ATTR
FUNCTION_NAME(
    PORTS(NUM_OF_ARGS, PORT))
{
    for (int i = 0; i < 10; i++)
        C[i] = A[i] + 1
}
#else
FUNCTION_TYPE
FUNCTION_NAME(
    PORTS(NUM_OF_ARGS, PORT))
{
    PORT_PRAGMAS(NUM_OF_ARGS, INTERFACE_PRAGMA)
#endif