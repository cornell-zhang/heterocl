[![CircleCI](https://circleci.com/gh/cornell-zhang/heterocl/tree/master.svg?style=svg)](https://circleci.com/gh/cornell-zhang/heterocl/tree/master)

HeteroCL: Heterogenous Computing Language
=========================================

[Installation](docs#installation-guide) | [Tutorial](docs/tutorial.md) | [API Documentation](docs#python-api) | 

## Introduction

High-level synthesis (HLS) gains its importance due to the rapid development of FPGA designs. However, to program a highly optimized design requires lots of expertise and experience. To conquer this, domain-specific languages (DSL) are developed to let users describe their designs in a more high-level fashion (such as tensor operations) and easily perform design space exploration (DSE). There are many existing DSLs, such as Halide and TVM. Nonetheless, most of the DSLs do not support imperative programming. Moreover, there is no clean code placement interface for efficient DSE. We propose heterogenous computing language (HeteroCL) to solve the above problems. HeteroCL is a DSL that combines both imperative and declarative programming and targets heterogeneous hardware devices, such as CPU, GPU, and FPGA. It also applies the programming style of decoupled function computation and scheduling, which is inspired by both Halide and TVM. Finally, it has a type system specifically designed for the heterogeneous system.

HeteroCL is similar to TVM in that it is an intermediate language lowered from other high-level DSL. However, it is also possible to program a design starting from HeteroCL. Unlike TVM, HeteroCL can describe non-tensor operations, such as control flow and data movement between different devices. After a HeteroCL code is generated/provided, it will be lowered to an intermediate representation (IR) for further optimization. HeteroCL IR is extended from Halide IR, which is again similar to TVM. The main difference is that we focus more on memory management, such as stencil analysis and the spatial architecture of memory. In addition, HeteroCL and its IR also support customized data types, such as fixed-point numbers, which are extremely important in hardware optimization. Finally, the code generation will take place with the optimized IR as its input. The type of generated code is based on the specified hardware device. It can be but is not limited to LLVM code, CUDA code, or HLS C code, for CPU, GPU, or FPGA execution, respectively.

<p align="center">
<img src="docs/Arch.png" width="250">
</p>

## Comparison with TVM

In general, for each TVM API, HeteroCL provides a corresponding wrapper. For example, one can declare a variable via `tvm.var` in TVM while `hcl.var` in HeteroCL. However, there exist some major differences between TVM and HeteroCL. Following lists only the major differences.

1. HeteroCL supports both imperative and declarative programming at DSL level.

   Similar to TVM, HeteroCL adapts vectorized code to describe tensor operations. With the vectorized code, the DSL becomes declarative since we only describe the results we want for each operation. Meanwhile, we use scheduling functions for the imperative part (i.e., how we'd like the functions to be implemented on devices). However, not every tensor operations can be vectorized elegantly. One example is sorting. We can definitely write something like
      ```python
      A = tvm.placeholder((10,), name = "A")
      B = tvm.compute(A.shape, lambda x: min(A, x), name = "B")
      # min(A, x) finds the x'th smallest value of A
      ```
   There exist two main problems with the above code snippet. First, the `min` function requires complex array operations or other data structures. Second, the computations involved are inefficient. For example, `min(A, 2)` requires `min(A, 1)`, which we have already computed. Although there are ways to reduce the redundancy, the best way is to write the sorting algorithm in an imperative fashion.

   TVM provides an API called `tvm.extern`, which allows users to describe a tensor operation (such as sorting) using provided IR interface. However, it is not intuitive for users to write programs at IR level. With HeteroCL, users can write simple Python code and it will be automatically lowered to IR. Another feature of HeteroCL is that users can further schedule the imperative code block. The `for` loops inside the code block are indexed according to their declaration order. All loop scheduling functions, such as unrolling and pipelining, can be applied. Following is an example of writing a sorting algorithm and scheduling it.

      ```python
      def insert_sort(A, B):
        for i in range(1, 10):
          for j in range(i, 0, -1):
            if A[j] > A[j-1]:
              swap(A[j], A[j-1])
        for k in range(0, 10):
          B[k] = A[k]
      
      A = hcl.placeholder((10,), name = "A")
      B = hcl.block([A], insert_sort, name = "B")
      
      s = hcl.create_schedule(B)
      s[B].pipeline(B.loops[0])
      ```

   In the above example, the first loop (with loop variable `i`) is pipelined and thus the second loop (with loop variable `j`) is fully unrolled, while the third loop (with loop variable `k`) remains the same.

2. HeteroCL provides a code placement interface.

   With TVM, users can specify on which the program is deployed. However, users cannot deploy parts of the program. With HeteroCL, users can decide (even at runtime) the code placement. Following is an example of code placement.

      ```python
      A = hcl.placeholder((10,), name = "A")
      B = hcl.compute(A.shape, lambda x: A[x] ^ 1, name = "B")
      C = hcl.compute(B.shape, lambda x: B[x] * A[x], name = "C")
      
      s = hcl.create_schedule(C)
      s[B].place(hcl.fpga)
      s[C].place(hcl.gpu)
      ```

   The default device for code placement is CPU. With heterogeneous systems, we need to consider the data movement between different devices, since it may become the bottleneck of the whole design. Users can also specify how the data movement should be done with HeteroCL. Following is an example.

      ```python
      # No content now
      ```
   <a name="dtype"></a>
3. HeteroCL supports customized data types.

   Customized data types are important for some hardware devices, such as FPGA. Some examples include `ap_int` and `ap_fixed`. TVM only supports common data types due to the limitation of Halide IR. With HeteroCL, the data type can be specified in the same way as TVM. Namely, specify the `dtype` field of a variable or a placeholder. Following lists the supported customized data type of HeteroCL.

   | Data Type | Description | Example |
   | :-------: | :------ | :-----: |
   | `int_{k}` | an integer with `k` bits, where `k` can be an arbitrary integer | `int_10` |
   | `uint_{k}` | an unsigned integer with `k` bits, where `k` can be an arbitrary integer | `uint_14` |
   | `float_{k}` | a floating point number with `k` bits, where `k` can be chosen from {8, 16, 32, 64} | `float_32` |
   | `fixed_{a}_{b}` | a fixed-point number with `a` bits, where the integer part has `b` bits | `fixed_10_2` |
   | `ufixed_{a}_{b}` | an unsigned fixed-point number with `a` bits, where the integer part has `b` bits | `ufixed_14_8` |

4. HeteroCL provides a type system for early error capturing before hardware synthesis.

   As mentioned in the second point, parts of the design can be deployed to different hardware devices. Thus, the data movement and communication between different devices are important and need to be dealt with carefully. Following is an example.

      ```python
      # Example required - usage of the same variable on different devices
      ```

   In addition, since we have customized data type, as mentioned in the third point, we can perform type checking for arithmetic operations. For example, HeteroCL gives a warning for potential overflow or underflow. Other warnings/errors include inconsistent array partition factor and loop unrolling factor, inconsistent numbers of data read/written from/to a FIFO, etc.
