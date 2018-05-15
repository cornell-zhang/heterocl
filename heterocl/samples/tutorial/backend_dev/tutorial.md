FPGA Backend Development Tutorial
=================================

<a name="top">List of Content</a>
-------
1. [Introduction](#intro)
2. [Create A Basic Code Generator](#basic)
3. [Register and Use the Code Generator](#register)
4. [Analyze AST for Code Generation](#analys)
-------

<h4>
1. <a name="intro">Introduction</a>
</h4>

In this tutorial, we are going to build a simple code generator for Xilinx HLS C step by step.
Note that we do not care about the performance in this tutorial. Instead, we focus on how to generate
a functional correct kernel.

<h4>
2. <a name="basic">Create A Basic Code Generator</a>
</h4>

We first create a basic code generator that generates pseudo C kernel, which is the code in
C-like syntax but cannot be compiled by gcc due to incomplete case processing.

We create a folder `tvm/src/codegen/hlsc/` and copy `codegen_hls.h` and `codegen_hls.cc` under `impl` to it.
These two files are the example code generator and can be modified to generate different languages.

Look into `codegen_hls.h`, there are three functions that we are interested in:
* `AddFunction` is the entry point of AST traversing.
* `Visit*` functions are the visitor functions for traversing AST nodes. The corresponding kernel code
will be printed to the string stream by each visitor.
* `Finish` is the result function that returns the kernel code in string type.

You can also review function implementations in `codege_hlsc.cc` to learn how each visitor function
writes the kernel code in C syntax.

Next, we copy `impl/build_hls.cc` to `tvm/src/codegen/hlsc`. This is the interface to the Python implementation.

Look into `build_hlsc.cc`, the function `TVM_REGISTER_API` is the function mapping to the Python API.
For example, `TVM_REGISTER_API("codegen.build_hlsc")` will create an API `build_hlsc` under `codegen` tvm codegen package.
In the function body, we call the function `BuildHLSC` and pass the `TVMArgs` as arguments. `TVMArgs` is an array of lowered
TVM functions, so we iteratively send the lowered function to `CodeGenHLSC.AddFunction` to generate the HLS kernel code.

In addition, as can be seen in `BuildHLSC`, a warning about lacking of runtime is specified. This because the original TVM
assumes every code generator has a corresponding host generator that is also invoked in this function to generate the host.
As a result, `BuildHLSC` should return a module with both host and kernel. However, HeteroCL targets to kernel generation only,
so `BuildHLSC` only return kernel code in string format as the output.

**IMPORTANT**: You **must** name this API as `codegen.build_target` where
`target` is the name you wish user to specify.

<h4>
3. <a name="register">Register and Use the code generator to the TVM Python interface</a>
</h4>

Finally, we specify the target kernel language `hlsc` in the TVM Python interface to expose to users.
To do so, we simply find the list named `FPGA_TARGETS` in `tvm/python/target.py` and add your backend
language name to the list. For example:

```python
FPGA_TARGETS = ['hlsc']
```

After that we can use the HeteroCL API to assign the target, as shown in `kernel.py`:

```python
f = hcl.build(s, [a, A, B], target='hls')
```

We can then generate the kernel by running the user program `python kernel.py` and get the following output:

```c
void default_function(uint5 arg0,  void* arg1,  void* arg2) {
  uint5 a = arg0;
  uint5* A = (uint5*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    CHECK((1 == ((int32_t)arg1_strides[0]))) << "arg1.strides: expected to be compact array";
  }
  int32_t dev_type = (((TVMArray*)arg1)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg1)[0].ctx.device_id);
  uint10* B = (uint10*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    CHECK((1 == ((int32_t)arg2_strides[0]))) << "arg2.strides: expected to be compact array";
  }
  for (int32_t x = 0; x < 10; ++x) {
    B[x] = (((uint10)A[x]) * ((uint10)a));
  }
}
```

Note that this is not the HLS C kernel (yet) but just a simple C-like pseudo code and cannot be compiled. In the next section,
we describe how to make the generated code legel for HLS C.
