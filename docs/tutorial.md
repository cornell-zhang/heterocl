Tutorial
========

List of Content<a name="top"></a>
-------
1. [Basic Usage](basic)
2. [Tensor Operations](op)
3. [Imperative Code Block](imp)
4. [Scheduling Functions](sch)
5. [Reduction Functions](red)
6. [Code Placement](code)
___

1. Basic Usage <a name="basic"></a>

   An HeteroCL variable or placeholder does not hold actual values. Their values will be evaluated at runtime. This is similar to `tvm.var` and `tvm.placeholder`. Following is an example.   
      ```python
      A = hcl.var("A", dtype = "int8")
      B = hcl.placeholder((10, 5), dtype = "int10", name = "B")
      ```
   Notice that for both HeteroCL variables and placeholders, we can specify their data types via the `dtype` field. The suppported types can be seen [here](README.md). After we have variables and placeholders, we can perform basic computations. Again, this is similar to `tvm.compute`. However, HeteroCL provides more options, which can be seen in the [next section](op). Following is an example of using `hcl.compute`.  
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
   Currently the `create_schedule` function is only used to build the module. We will introduce more in the later sections. In the `build` function, we specify which variables/tensors are the inputs/outputs of the program. The inputs/outputs to the program can be generated from `numpy` arrays. The data type of the arrays will be handled by HeteroCL. Similarly, a warning message will be given if there exist a type mismatch.
