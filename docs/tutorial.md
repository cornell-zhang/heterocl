Tutorial
========

<a name="top">List of Content</a>
-------
1. [Getting Started](#basic)
2. [Imperative Code Block](#imp)
3. [Tensor Operations](#op)
4. [Scheduling Functions](#sch)
5. [Bit-Accurate Data Types](#badt)
6. [Reduction Functions](#red)
7. [Code Placement](#code)
8. [Basic Optimization](#opt)
___

1. <a name="basic">Getting Started</a>

   This is an introduction to the basic usage of HeteroCL. By finishing this section, you will have the ability to write a simple HeteroCL program. HeteroCL is a programming model based on [TVM](http://tvmlang.org). For TVM users, HeteroCL should be easy to learn. However, it is totally fine if you are not familar with TVM.
   
   To begin with, a HeteroCL program consists of input variables, intermediate variables, and algorithm description. Each variable in HeteroCL can be either a *Var* or a *Tensor*. They serve as containers to hold values. Users can assign values to/retrieve values from them during runtime. This conecpt is also used in many popular domain-specific languages, such as Halide and TensorFlow.
   
   A Var is a scalar and is **immutable**. Namely, you cannot change the value of a Var. Thus, a Var can only be an input to the program. On the other hand, a Tensor is **mutable**. You can get/set the value of an element in the tensor. This means that any Tensor in HeteroCL can be an output to the program. To create a mutable scalar, you can create a one-dimensional Tensor with exactly one element. Following shows how to declare a Var and a Tensor with no computation involved.   
      ```python
      import heterocl as hcl
      a = hcl.var(name, dtype) # returns a Var
      A = hcl.placeholder(shape, name, dtype) # returns a Tensor
      
      # example
      a = hcl.var("a")
      A = hcl.placeholder((10, 10), "A", hcl.Int(10))
      ```
   For each variable in HeteroCL, we can specify the name of it. If not, a name will be automatically generated. We can also specify its data type via the `dtype` field. The possible data types are listed below.
      ```python
      hcl.UInt(k)        # a k-bit unsigned integer; k is 32 if not specified
      hcl.Int(k)         # a k-bit signed integer; k is 32 if not specified
      hcl.UFixed(k, l)   # a k-bit unsigned fixed-point with l fractional bits
      hcl.Fixed(k, l)    # a k-bit signed fixed-point with l fractional bits
      hcl.Float()        # a floating-point (always 32-bit)
      hcl.Double()       # a double-precision floating-point (always 64-bit)
      ```
   For more details on bit-accurate data types, check [this section](#badt). **The default data type for all variables is `Int(32)`.** You can set the default data type by setting `config.init_dtype`.
      ```python
      hcl.config.init_dtype = Float()
      ```
   One important concept of HeteroCL is *Stage*. A set of Stages can connect to each other and form a data flow graph. For `var` and `placeholder`, they serve as input Stages. In other words, there is no preceding Stage for a `var` or a `placeholder`.
   > The returned object of a HeteroCL API can have multiple roles. For example, the returned object of `hcl.var` is not only a Var but also a Stage.
   
   After we have the input stages, we can now describe our algorithms. HeteroCL provides several ways to describe an algorithm. One way is to use vectorized code to describe tensor operations. The first computation related API we are going to introduce is `hcl.compute`. Following shows the definition and the effect of the API.
      ```python
      B = hcl.copmute(shape, inputs, fcompute, name, dtype) # returns a Tensor
      
      # B[index] = fcompute(index), for all index in shape
      ```
   This API takes in the output shape, a list of input stages, a lambda function with indices as arguments, a name and data type. The last two arguments are optional. Following shows a code snippet from [Example 1](/heterocl/samples/tutorial/example_1.py).
      ```python
      a = hcl.var()
      A = hcl.placeholder((10, 10))
      B = hcl.compute((10, 10), [A], lambda x, y: A[x][y] * a)
      
      # equivalent Python code
      for x in range(10):
        for y in range(10):
          B[x][y] = A[x][y] * a
      ```
   Note that for the input list, we do not need to include Vars. For other algorithm description, please refer to [this section](#op). Now we can build the above program. First, we create a schedule for the whole program using `hcl.create_schedule`. Any Stage in the program can be scheduled. In this section we are not going to further discuss scheduling. Please look at [Scheduling Functions](#sch) for more information. To create a schedule, we need the last Stage. Following shows the API of `hcl.create_schedule`.
      ```python
      s = hcl.create_schedule(op)
      ```
   Finally, we can build the program via `hcl.build`, which takes in the schedule we just created and all input/output variables.
      ```python
      f = hcl.build(schedule, in_outs)
      ```
   The return value is a function that takes in the specified inputs and outputs in the form of HeteroCL arrays, which can be created using Numpy arrays. To do so, we can use `hcl.asarray`.
      ```python
      hcl_arr = hcl.asarray(numpy_arr, dtype)
      ```
   The `dtype` must match the data type of the corresponding variable. We can also transform a HeteroCL array back to a Numpy array using `asnumpy`.
      ```python
      numpy_arr = hcl_arr.asnumpy()
      ```
   Here we show the complete code of [Example 1](/heterocl/samples/tutorial/example_1.py).
      ```python
      a = hcl.var()
      A = hcl.placeholder((10, 10))
      B = hcl.compute((10, 10), [A], lambda x, y: A[x, y] * a)
      
      s = hcl.create_schedule(B)
      f = hcl.build(s, [a, A, B])
      
      hcl_a = 10
      hcl_A = hcl.asarray(numpy.random.randint(100, size = (10, 10)), dtype = hcl.UInt())
      hcl_B = hcl.asarray(numpy.zeros((10, 10)), dtype = hcl.UInt())
      
      f(hcl_a, hcl_A, hcl_B)
      
      print hcl_a
      print hcl_A.asnumpy()
      print hcl_B.asnumpy()
      ```
   <p align="right"><a href="#top">↥</a></p>

2. <a name="imp">Imperative Code Block</a>

   One major feature of HeteroCL is that it allows users to write mixed-imperative-declarative programs. 
      ```python
      # Example 2
      def popcount(a): # a is a 32-bit integer
        out = hcl.local(0)
        with hcl.for_(0, 32) as i:
          out[0] += a[i]
        return out[0]
      
      A = hcl.placeholder((10, 10))
      B = hcl.compute((10, 10), [A], lambda x, y: popcount(A[x, y]))
      ```
   In the above [example](/heterocl/samples/tutorial/example_2.py), we calculate the popcount of each number in tensor `A` and store the value in tensor `B`. There are many ways we can write a popcount algorithm. Here we just use this example to show how to write imperative code in HeteroCL. Nothe that **imperative code can only be used inside HeteroCL APIs**.
   
   HeteroCL provides a DSL for writing imperative codes. Following shows the definition of HeteroCL DSL.
      ```python
      expr := Var | Tensor[expr] | Number | Boolean 
              | Unary_Op | Binary_Op | Comp_Op |
              | expr[expr] # get bit
              | expr[expr:expr] # get slice
      stmt := Tensor[expr] = expr
              | expr[expr] = expr # set bit
              | expr[expr:expr] = expr # set slice
              | with hcl.if_(expr):
                  stmt
              | with hcl.else_(): # cannot use without hcl.if_
                  stmt
              | with hcl.for_(expr, expr) as var:
                  stmt
              | HeteroCL APIs (e.g., hcl.local, hcl.compute, etc.)
      Unary_Op := not expr
      Bin_Op := expr op expr, where op ∈ [+, -, *, /, %, &, |, ^]
      Comp_Op := expr op expr, where op ∈ [>, >=, ==, <=, <, !=]
      ```
   To include a pure imperative code block, we can use `hcl.block`.
      ```python
      A = hcl.block(input_stages, fblock)
      ```
   The return value `A` is a Stage. With this new API, we can alternatively rewrite [Example 2](/heterocl/samples/tutorial/example_2.py) as shown here.
      ```python
      def popcount(A, B): # a is a 32-bit integer
          with hcl.for_(0, 10) as x:
            with hcl.for_(0, 10) as y:
              B[x, y] = 0
              with hcl.for_(0, 32) as i:
                B[x, y] += A[x, y][i]
      
      A = hcl.placeholder((10, 10))
      B = hcl.placeholder((10, 10))
      C = hcl.block([A, B], lambda: popcount(A, B))
      ```
   <p align="right"><a href="#top">↥</a></p>
   
3. <a name="op">Tensor Operations</a>

   In addition to `hcl.compute` and `hcl.block`, HeteroCL supports more tensor operations. The first one is `hcl.update`. HeteroCL supports tensor **in-place** update. Namely, no new tensor is returned.
      ```python
      A = hcl.update(target, inputs, fupdate)
      
      # target[indices] = fupdate(indices)
      ```
   The target tensor must appear in the list of inputs. Since we are updating the target in place, we need to write `fupdate` carefully. Otherwise, we might end up with incorrect behavior. HeteroCL also supports another tensor operation called `hcl.mut_compute`. Users can use this API to vectorize any loops.
      ```python
      A = hcl.mut_compute(domain, inputs, fcompute)
      ```
   In this API, `fcompute` will be executed for each index in the given `domain`. Following is an example.
      ```python
      def shift_reg(A, x):
        with hcl.CodeBuilder() as cb:
          with hcl._for(0, 10) as k:
            A[k] = A[k+1]
            
      A = hcl.placeholder((10,))
      B = hcl.mut_compute((5,), [A], lambda x: shift_reg(A, x))
      
      # equivalent code
      for x in range(5):
        for k in range(9):
          A[k] = A[k+1]
        A[9] = 0
      ```
   In the above example, we shift tensor `A` to the left for five times. This API enables users to write more powerful vector code.
   <p align="right"><a href="#top">↥</a></p>

4. <a name="sch">Scheduling Functions</a>

   <p align="right"><a href="#top">↥</a></p>
