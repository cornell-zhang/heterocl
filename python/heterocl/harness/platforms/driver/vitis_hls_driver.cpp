#pragma message("driver code for Vitis HLS backend")

// Enable test and run "gcc -E vitis_hls_driver.cpp"
#define TEST

// ===== HLS top function information =====
#ifdef TEST
#define FUNCTION_NAME top
#define NUM_OF_ARGS 2

#define argType0 int *
#define argName0 A
#define argDim0
#define argParam0 mode = m_axi bundle = gmem0

#define argType1 int
#define argName1 C
#define argDim1 [10]
#define argParam1 mode = m_axi bundle = gmem1
#endif
// ====== End of configurable knobs =====

// HLS include headers
#ifndef TEST
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#endif

// macro utility functions
#define DO_PRAGMA(x) _Pragma(#x)
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

#define DUP_PRAGMAS(n, c) DUP_PRAGMAS##n(c)
#define DUP_PRAGMAS7(c) c(argName0, argParam0) c(argName1, argParam1) c(argName2, argParam2) c(argName3, argParam3) \
    c(argName4, argParam4) c(argName5, argParam5) c(argName6, argParam6)
#define DUP_PRAGMAS6(c) c(argName0, argParam0) c(argName1, argParam1) c(argName2, argParam2) c(argName3, argParam3) \
    c(argName4, argParam4) c(argName5, argParam5)
#define DUP_PRAGMAS5(c) c(argName0, argParam0) c(argName1, argParam1) c(argName2, argParam2) c(argName3, argParam3) \
    c(argName4, argParam4)
#define DUP_PRAGMAS4(c) c(argName0, argParam0) c(argName1, argParam1) c(argName2, argParam2) c(argName3, argParam3)
#define DUP_PRAGMAS3(c) c(argName0, argParam0) c(argName1, argParam1) c(argName2, argParam2)
#define DUP_PRAGMAS2(c) c(argName0, argParam0) c(argName1, argParam1)
#define DUP_PRAGMAS1(c) c(argName0, argParam0)
#define DUP_PRAGMAS0(c)

#define PORTS(num, c) DUP_ARGS(num, c)
#define PORT_PRAGMAS(num, c) DUP_PRAGMAS(num, c)

#ifdef TEST
// define function name and ports
void FUNCTION_NAME(
    PORTS(NUM_OF_ARGS, PORT))
{
    PORT_PRAGMAS(NUM_OF_ARGS, INTERFACE_PRAGMA)
    for (int i = 0; i < 10; i++)
        C[i] = A[i] + 1
}
#else
void FUNCTION_NAME(
    PORTS(NUM_OF_ARGS, PORT))
{
    PORT_PRAGMAS(NUM_OF_ARGS, INTERFACE_PRAGMA)
#endif