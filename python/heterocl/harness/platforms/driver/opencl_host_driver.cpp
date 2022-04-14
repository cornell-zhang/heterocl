#define TEST

#include <cstdlib>
#include <iostream>
// #include "frt.h"

#ifdef TEST
#define NUM_OF_ARGS 2
#define BITSTREAM "kernel.xclbin"

#define argType0 int
#define argName0 A
#define argDim0 10
#define argAccess0 fpga::ReadOnly

#define argType1 int
#define argName1 C
#define argDim1 10
#define argAccess1 fpga::WriteOnly
#endif

#define PORT(argType, argName, argDim, argAccess) argAccess(argName, sizeof(argType) * argDim)
#define DUP_ARGS(n, c) DUP_ARGS##n(c)
#define DUP_ARGS2(c) c(argType0, argName0, argDim0, argAccess0) c(argType1, argName1, argDim1, argAccess1)
#define DUP_ARGS1(c) c(argType0, argName0, argDim0, argAccess0)
#define DUP_ARGS0(c)
#define PORTS(num, c) DUP_ARGS(num, c)

#define PRINT_ARG_INFO                             \
    for (const auto &arg : instance.GetArgsInfo()) \
        clog << arg << "\n";

#define PRINT_PERF_NUM                                                                     \
    clog << "[+] Load throughput: " << instance.LoadThroughputGbps() << " GB/s\n";         \
    clog << "[+] Compute latency: " << instance.ComputeTimeSeconds() << " s" << std::endl; \
    clog << "[+] Store throughput: " << instance.StoreThroughputGbps() << " GB/s\n";

#ifdef TEST
int main(int argc, char *argv[])
{

    auto instance = fpga::Invoke(BITSTREAM,
                                 PORTS(NUM_OF_ARGS, PORT));
    PRINT_ARG_INFO
    PRINT_PERF_NUM
}
#else
int main(int argc, char *argv[])
{

    auto instance = fpga::Invoke(BITSTREAM,
                                 PORTS(NUM_OF_ARGS, PORT));
#endif