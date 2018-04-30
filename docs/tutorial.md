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

   This is an introduction to the basic usage of HeteroCL. HeteroCL is a general-purpose programming model for heterogenous backends. To begin with, each variable in HeteroCL can be either a **var** or a **placeholder**. They serve as containers to hold values. To retrieve the values, users need to evaluate them at runtime. Following are the definitions.   
      ```python
      a = hcl.var(name, dtype)
      A = hcl.placeholder(shape, name, dtype)
      ```
   For each variable in HeteroCL, we can specify the name of it, which is useful during debugging. We can also specify its data type via the `dtype` field. You can also specify the data type later using `hcl.resize`. **The default data type for all variables is `int32`.** After we have the variables, we can now describe our algorithms. HeteroCL provides several ways to describe an algorithm. One way is to use vectorized code to describe tensor operations. The first API we are going to introduce is `hcl.compute`. Following shows the definition and the effect of the API.
      ```python
      B = hcl.copmute(shape, inputs, fcompute, name, dtype)
      
      # B[index] = fcompute(index), for all index in shape
      ```
   This API takes in the output shape, a list of inputs, a lambda function with indices as arguments, and a name and data type. The last two arguments are optional. This is an example. <a name="ex1"></a>
      ```python
      # Example 1
      a = hcl.var()
      A = hcl.placeholder((10, 10))
      B = hcl.compute((10, 10), [A], lambda x, y: A[x][y] * a)
      
      # equivalent Python code
      for x in range(10):
        for y in range(10):
          B[x][y] = A[x][y] * a
      ```
   Note that for the input list, we only need to include tensors. Now we can build the above program. First, we can specify the data types of the variables by using `hcl.resize`. The definition is shown below.
      ```python
      hcl.resize(variables, dtype)
      ```
   After that, we create a schedule for the whole program using `hcl.create_schedule`. Please look at [Scheduling Functions](#sch) for more information. To create a schedule, we need the last operation. Most tensor operation APIs will return an operation although in some cases no new tensor is created. For the list of tensor operations, please refer to [Tensor Operations](#op). Following shows the API of `hcl.create_schedule`.
      ```python
      s = hcl.create_schedule(op)
      ```
   Finally, we can build the program via `hcl.build`, which takes in the schedule we just created and all input/output variables.
      ```python
      f = hcl.build(schedule, in_outs)
      ```
   The return value is a function that takes in the specified inputs and outputs, which can be created using Numpy arrays. To create an input/output tensor, we can use `hcl.asarray`. We can also transform an HeteroCL tensor back to a Numpy array using `hcl.asnumpy`. Here we show the complete code of Example 1.
      ```python
      a = hcl.var()
      A = hcl.placeholder((10, 10))
      B = hcl.compute((10, 10), [A], lambda x, y: A[x, y] * a)
      
      hcl.resize([a, A, B], "uint6")
      
      s = hcl.create_schedule(B)
      f = hcl.build(s, [a, A, B])
      
      hcl_a = 10
      hcl_A = hcl.asarray(numpy.random.randint(100, size = (10, 10)), dtype = "uint6)
      hcl_B = hcl.asarray(numpy.zeros((10, 10)), dtype = "uint6")
      
      f(hcl_a, hcl_A, hcl_B)
      
      print hcl_B.asnumpy()
      ```
   Since we quantize each variable to 6-bit, the results should be no larger than 63.
   <p align="right"><a href="#top">↥</a></p>

2. <a name="op">Imperative Code Block</a>

   This is another feature of HeteroCL. An imperative code block can appear in most HeteroCL APIs. We can see an example for how it works.
      ```python
      # Example 2
      def popcount(a): # a is a 32-bit integer
        with hcl.CodeBuilder() as cb:
          out = hcl.local(0)
          with cb._for(0, 32) as i:
            out[0] += a[i]
          return out[0]
      
      A = hcl.placeholder((10, 10))
      B = hcl.compute((10, 10), [A], lambda x, y: popcount(A[x, y]))
      ```
   In the above example, we calculate the popcount of each number in tensor `A` and store the value in tensor `B`. To write an imperative code block, we first need a `CodeBuilder`. In the body of a `CodeBuilder`, we can declare variables, assign values to tensors, use any HeteroCL API, call functions, and program control flows. Following shows a complete list of what can be done inside an imperative code block.
      ```python
      with hcl.CodeBuilder() as cb:
        # Variables declaration
        a = hcl.local(init) # syntatic sugar for hcl.compute((1,), [], lambda x: init)
        b = hcl.var(...)
        C = hcl.placeholder(...)
        D = hcl.compute(...)
        
        # Tesnor assignments
        a[0] = C[2, 3]
        
        # Use HeteroCL APIs (not limited to these)
        r = hcl.reduce_axis(...)
        R = hcl.reducer(...)
        
        # Function call (use the same CodeBuilder in the function)
        f(..., cb)
        
        # For loops
        with cb._for(_min, _max) as loop_var:
          # imperative body
          
        # If/Else statement (Else is not necessary)
        with cb._if(cond):
          # imperative body
        with cb._else():
          # imperative body
      ```
   To include a pure imperative code block, we can make use of `hcl.block`. This is how it looks like.
      ```python
      A = hcl.block(inputs, fblock)
      ```
   The return value `A` is an operation that can be used during scheduling. We can rewrite Example 2 using `hcl.block`.
      ```python
      def popcount(A, B): # a is a 32-bit integer
        with hcl.CodeBuilder() as cb:
          with cb._for(0, 10) as x:
            with cb._for(0, 10) as y:
              B[x, y] = 0
              with cb._for(0, 32) as i:
                B[x, y] += A[x, y][i]
      
      A = hcl.placeholder((10, 10))
      B = hcl.placeholder((10, 10))
      C = hcl.block([A, B], lambda a, b: popcount(a, b))
      ```
   In the above code snippet, the arguments of the `lambda` function correspond to the this of inputs. Also, we can see that there is no longer a return value in `popcount`. The values of tensor `B` is updated inside `popcount`.
   <p align="right"><a href="#top">↥</a></p>
   
3. <a name="imp">Tensor Operations</a>

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
