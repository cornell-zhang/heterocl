[![CircleCI](https://circleci.com/gh/cornell-zhang/heterocl/tree/master.svg?style=svg&circle-token=2b5ee9faf30b94aac41b61032d03e4654a65079d)](https://circleci.com/gh/cornell-zhang/heterocl/tree/master)

HeteroCL: Heterogenous Computing Language
=========================================

[Installation](docs#installation-guide) | [Tutorial](docs/tutorial.md) | [Documentation](http://people.ece.cornell.edu/yl2666/heterocl/docs/build/html/)

## Introduction

High-level synthesis (HLS) gains its importance due to the rapid development of FPGA designs. However, to program a highly optimized design requires lots of expertise and experience. To conquer this, domain-specific languages (DSL) are developed to let users describe their designs in a more high-level fashion (such as tensor operations) and easily perform design space exploration (DSE). There are many existing DSLs, such as Halide and TVM. Nonetheless, most of the DSLs do not support imperative programming. Moreover, there is no clean code placement interface for efficient DSE. We propose heterogenous computing language (HeteroCL) to solve the above problems. HeteroCL is a DSL that combines both imperative and declarative programming and targets heterogeneous hardware devices, such as CPU, GPU, and FPGA. It also applies the programming style of decoupled function computation and scheduling, which is inspired by both Halide and TVM. Finally, it has a type system specifically designed for the heterogeneous system.

HeteroCL is similar to TVM in that it is an intermediate language lowered from other high-level DSL. However, it is also possible to program a design starting from HeteroCL. Unlike TVM, HeteroCL can describe non-tensor operations, such as control flow and data movement between different devices. After a HeteroCL code is generated/provided, it will be lowered to an intermediate representation (IR) for further optimization. HeteroCL IR is extended from Halide IR, which is again similar to TVM. The main difference is that we focus more on memory management, such as stencil analysis and the spatial architecture of memory. In addition, HeteroCL and its IR also support customized data types, such as fixed-point numbers, which are extremely important in hardware optimization. Finally, the code generation will take place with the optimized IR as its input. The type of generated code is based on the specified hardware device. It can be but is not limited to LLVM code, CUDA code, or HLS C code, for CPU, GPU, or FPGA execution, respectively.

<p align="center">
<img src="docs/Arch.png" width="250">
</p>

## Comparison with TVM
