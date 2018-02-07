Tutorial
========

<a name="top">List of Content</a>
-------
1. [Basic Usage](#basic)
2. [Tensor Operations](#op)
3. [Imperative Code Block](#imp)
4. [Scheduling Functions](#sch)
5. [Reduction Functions](#red)
6. [Code Placement](#code)
7. [Basic Optimization](#opt)
___

1. <a name="basic">Basic Usage</a>

   An HeteroCL variable or placeholder does not hold actual values. Their values will be evaluated at runtime. This is similar to `tvm.var` and `tvm.placeholder`. Following is an example.   
      ```python
      A = hcl.var("A", dtype = "int8")
      B = hcl.placeholder((10, 5), dtype = "int10", name = "B")
      ```
   Notice that for both HeteroCL variables and placeholders, we can specify their data types via the `dtype` field. The suppported types can be seen [here](../README.md#dtype). After we have variables and placeholders, we can perform basic computations. Again, this is similar to `tvm.compute`. However, HeteroCL provides more options, which can be seen in the [next section](#op). Following is an example of using `hcl.compute`.  
      ```python
      C = hcl.compute(B.shape, lambda x, y: B[x, y] + A, name = "C")
      ```
   The above example is equivalent to the following Python program.
      ```python
      for x in range(0, 10):
        for y in range(0, 5):
          C[x][y] = B[x][y] + A
      ```
   Also, we do not need to specify the output data type, which will be automatically inferred. However, we can specify it manually. The HeteroCL type system will give a warning if there exist a type mismatch. Now we have the whole program. The following code snippet shows how we build the program and run it.
      ```python
      s = hcl.create_schedule(C)
      m = hcl.build(s, [A, B, C])
      
      a = 10
      b = hcl.asarray(numpy.random.randint(1024, size = (10, 5)))
      c = hcl.asarray(numpy.zeros(10, 5))
      m(a, b, c)
      print c.asnumpy()
      ```
   Currently the `create_schedule` function is only used to build the module. We will introduce more in the later sections. In the `build` function, we specify which variables/tensors are the inputs/outputs of the program. The inputs/outputs to the program can be generated from `numpy` arrays. The data type of the arrays will be handled by HeteroCL. Similarly, a warning message will be given if there exists a type mismatch.
   <p align="right"><a href="#top">↥</a></p>

2. <a name="op">Tensor Operations</a>

   From the [previous section](#basic), we learned that `hcl.copmute` can be used to perform tensor operations. HeteroCL provides more options for tensor operations. In TVM, every operation inside the `lambda` function will be inlined. Following is an example.
      ```python
      def foo(A, x):
        b = 0
        for i in range(0, 3):
          b += A[x+1-i]
        return b
      
      A = tvm.placeholder((10,), name = "A")
      B = tvm.compute(A.shape, lambda x: foo(A, x), name = "B")
      ```
   With the above example, we will get this after lowering.
      ```python
      for x in range(0, 10):
        B[x] = A[x+1] + A[x] + A[x-1]
      ```
   Namely, the loop inside `foo` is inlined. With HeteroCL, users can decide whether it should be inlinede or not via the `ineline` option. We can rewrite the above program using HeteroCL with the `inline` option.
      ```python
      B = hcl.compute(A.shape, lambda x: foo(A, x), name = "B", inline = False)
      ```
   And after lowering, we get
      ```python
      for x in range(0, 10):
        b = 0
        for i in range(0, 3):
          b += A[x+1-i]
        B[x] = b
      ```
   With this option, users can later on use scheduling functions, such as `inline` or `unroll`, for design space exploration.
   > By default, the inline option is set to `True`. Moreover, the condition statements, such as `if`, will also be inlined/evaluated. To prevent from being inlined/evaluted, users can use `hcl.select`. Examples of using `hcl.select` can be seen [here](docs/api.md).
   
   HeteroCL also provides a new API called `update`, which allows users to update the same placeholder without creating a new one. Following is an example.
      ```python
      A = hcl.placeholder((10,), name = "A")
      B = hcl.compute(A.shape, lambda x: A[x] + 1, name = "B")
      B1 = hcl.update(B, lambda x: B[x] * 5, name = "B1")
      # after lowering, we get
      for x in range(0, 10):
        B[x] = A[x] + 1
      for x in range(0, 10):
        B[x] = B[x] * 5
      ```
   Note that `B1` is not a placeholder since we do not create any new one. It will be used for scheduling. This API is useful if we have limited harware resources.
   <p align="right"><a href="#top">↥</a></p>
   
3. <a name="imp">Imperative Code Block</a>

   One main feature of HeteroCL is that it provides an API for writing an imperative code block. To write it, one only needs to wrap the code into a python function and use the `hcl.block` interface.
      ```python
      def popcount(inputs, outputs, *args):
        A = inputs[0]
        B = outputs[0]
        length = args[0]
        for x in range(0, 10):
          B[x] = 0
          for y in range(0, length):
            if A[x] % 2 == 1:
              B[x] += 1
            A[x] = A[x] >> 1
      
      A = hcl.placeholder((10,), name = "A", dtype = "int49")
      B = hcl.placeholder((10,), name = "B", dtype = "int6")
      C = hcl.block([A], [B], [49], popcount)
      ```
   In the above example, we are computing the popcount (i.e., how many ones for a number in its binary representation) for each element of `A` and storing the result in `B`. First, the input fields for `hcl.block` are the list of inputs, the list of outputs, the list of other arguments, and the imperative function, respectively. For the function, its argument **must be** in the form of inputs, outputs, and arguments. Otherwise, the users will get an error message. Similarly, `C` is used for scheduling.

