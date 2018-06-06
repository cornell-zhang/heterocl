Tutorial
========

<a name="top">List of Content</a>
-------
1. [Getting Started](#basic)
2. [Imperative Code Block](#imp)
3. [More Algorithm Description APIs](#op)
4. [Bit-Accurate Data Types](#badt)
5. [Function Block](#func)
6. [Reduction Operations](#red)
7. [Scheduling Functions](#sch)
8. [Code Placement](#code)
9. [Basic Optimization](#opt)
___

## 1. <a name="basic">Getting Started</a>

This is an introduction to the basic usage of HeteroCL. By finishing this section, you will have the ability to write a simple HeteroCL program. HeteroCL is a programming model based on [TVM](http://tvmlang.org). For TVM users, HeteroCL should be easy to learn. However, it is totally fine if you are not familiar with TVM.
   
To begin with, a HeteroCL program consists of input variables, intermediate variables, and algorithm description. Each variable in HeteroCL can be either a *Var* or a *Tensor*. They serve as containers to hold values. Users can assign values to/retrieve values from them during runtime. This concept is also used in many popular domain-specific languages, such as Halide and TensorFlow.
   
A Var is a scalar and is **immutable**. Namely, you cannot change the value of a Var. Thus, a Var can only be an input to the program. On the other hand, a Tensor is **mutable**. You can get/set the value of an element in the tensor, which means that any Tensor in HeteroCL can be an output to the program. To create a mutable scalar, you can create a one-dimensional Tensor with one element. Following shows how to declare a Var and a Tensor with no computation involved.
   
   ```python
   import heterocl as hcl
   a = hcl.var(name, dtype) # returns a Var
   A = hcl.placeholder(shape, name, dtype) # returns a Tensor
        
   # example
   a = hcl.var("a")
   A = hcl.placeholder((10, 10), "A", hcl.Int(10))
   ```

For each variable in HeteroCL, we can specify the name of it. If not, a name will be automatically generated. We can also specify its data type via the `dtype` field. The possible data types are listed below. <a name="dt_table"></a>
      
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
   
A critical concept of HeteroCL is *Stage*. A set of Stages can connect to each other and form a data flow graph. For `var` and `placeholder`, they serve as input Stages. In other words, there is no preceding Stage for a `var` or a `placeholder`.

> The returned object of a HeteroCL API can have multiple roles. For example, the returned object of `hcl.var` is not only a Var but also a Stage.
   
After we have the input stages, we can now describe our algorithms. HeteroCL provides several ways to express an algorithm. One way is to use vectorized code to describe tensor operations. The first computation related API we are going to introduce is `hcl.compute`. Following shows the definition and the effect of the API.
      
   ```python
   B = hcl.copmute(shape, inputs, fcompute, name, dtype) # returns a Tensor
      
   # B[index] = fcompute(index), for all index in shape
   ```
   
This API takes in the output shape, a list of input stages, a lambda function with indices as arguments, a name and data type. The last two arguments are optional. **The length of the output shape must match the number of lambda arguments.** Following shows a code snippet from [Example 1](/heterocl/samples/tutorial/example_1.py).
      
   ```python
   a = hcl.var()
   A = hcl.placeholder((10, 10))
   B = hcl.compute((10, 10), [A], lambda x, y: A[x][y] * a)
      
   # equivalent Python code
   for x in range(10):
     for y in range(10):
       B[x][y] = A[x][y] * a
   ```
   
Note that for the input list, we do not need to include Vars. For other algorithm description, please refer to [this section](#op). Now we can build the above program. First, we create a schedule for the whole program using `hcl.create_schedule`. Any Stage in the program can be scheduled. In this section, we are not going to discuss scheduling. Please look at [Scheduling Functions](#sch) for more information. To create a schedule, we need the last Stage(s). Following shows the API of `hcl.create_schedule`.
      
   ```python
   s = hcl.create_schedule(ops)
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
   B = hcl.compute(A.shape, [A], lambda x, y: A[x, y] * a)
      
   s = hcl.create_schedule(B)
   f = hcl.build(s, [a, A, B])
      
   hcl_a = 10
   hcl_A = hcl.asarray(numpy.random.randint(100, size = (10, 10)), dtype = hcl.Int())
   hcl_B = hcl.asarray(numpy.zeros((10, 10)), dtype = hcl.Int())
      
   f(hcl_a, hcl_A, hcl_B)
      
   print hcl_a
   print hcl_A.asnumpy()
   print hcl_B.asnumpy()
   ```

You can also inspect the generated IR using `hcl.lower`. The usage is the same as that of `hcl.build`, where the arguments include the schedule and a list of variables.

<p align="right"><a href="#top">↥</a></p>

## 2. <a name="imp">Imperative Code Block</a>

One significant feature of HeteroCL is that it allows users to write mixed-imperative-declarative programs. 
   
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
   
In the above [example](/heterocl/samples/tutorial/example_2.py), we calculate the popcount of each number in tensor `A` and store the value in tensor `B`. There are many ways we can write a popcount algorithm. Here we use this example to show how to write imperative code in HeteroCL. Note that **imperative code can only be used inside HeteroCL APIs**.

### 2.1 <a name="dsl">HeteroCL DSL for Imperative Code</a>

HeteroCL provides a DSL for writing imperative code. Following shows the definition of HeteroCL DSL.
      
   ```python
   BinOp := [+, -, *, /, %, &, |, ^, >>, <<]
   BinEqOp := [+=, -=, *=, /=]
   CompOp := [>, >=, ==, <=, <, !=]
   expr := Var | Tensor[expr] | Number |
           | not expr | expr BinOp expr |
           | expr[expr] # get bit
           | expr[expr:expr] # get slice
   cond := expr CompOp expr | hcl.and_(*cond) | hcl.or_(*cond)
   stmt := Tensor[expr] = expr | Tensor[expr] BinEqOp expr
           | expr[expr] = expr # set bit
           | expr[expr:expr] = expr # set slice
           | with hcl.if_(cond):
               stmt
           | with hcl.elif_(cond): # cannot use without hcl.if_ or hcl.elif_
               stmt
           | with hcl.else_(): # cannot use without hcl.if_ or hcl.elif_
               stmt
           | with hcl.for_(expr, expr) as Var:
               stmt
           | with hcl.while_(cond):
               stmt
           | hcl.break_() # must be used inside hcl.for_ or hcl.while_
           | HeteroCL APIs (e.g., hcl.compute)
   ```
   
To include an imperative code block, we can use `hcl.block`.
      
   ```python
   A = hcl.block(input_stages, fblock)
   ```

The return value `A` is a Stage. With this new API, we can alternatively rewrite [Example 2](/heterocl/samples/tutorial/example_2.py) as shown here.
      
   ```python
   def popcount(A, B): # a is a 32-bit integer
     with hcl.for_(0, A.shape[0]) as x:
       with hcl.for_(0, A.shape[1]) as y:
         B[x, y] = 0
           with hcl.for_(0, 32) as i:
             B[x, y] += A[x, y][i]
      
   A = hcl.placeholder((10, 10))
   B = hcl.placeholder(A.shape)
   C = hcl.block([A, B], lambda: popcount(A, B))
      
   s = hcl.create_schedule(C)
   ``` 
   
> It is not allowed to have `hcl.block` inside another `hcl.block`. However, you can call a Python function inside `hcl.block` which is also written in imperative DSL.
   
Note that HeteroCL does not support scalars. However, it provides a similar API called `hcl.local`, which creates a 1D tensor with one element. The input is the initial value.
      
   ```python
   a = hcl.local(init_val, name, dtype)
   ```

Nonetheless, you need to be careful when using the returned object `a` because it is still a tensor. Thus, when you access the value (either reading or writing), you need to use `a[0]`.
   
### 2.2. <a name="vimp">Accessing variables inside an imperative code block</a>

HeteroCL provides a way for users to access variables inside a code block, which can be useful in the later sections such as scheduling. The only thing you need to do is to name the variables inside the code block. After that, you can access them as properties to the corresponding stage. More details will be covered in the later sections. Following shows an example.

   ```python
   def myfunc(A):
     l = hcl.local(0, "l")
     
     def myfoo(x):
       li = hcl.local(x, "li")
       return li[0] + l[0]
     
     with hcl.for_(0, A.shape[0], "x") as x:
       with hcl.for_(0, A.shape[1], "y") as y:
         A[x, y] = l[0] + 1
     hcl.update(A, [A, l], lambda x, y: myfoo(x), "U")
   
   A = hcl.placeholder((10, 10))
   B = hcl.block([A], lambda: myfunc(A))
   
   s = hcl.create_schedule(B)
   
   # use these after scheduling
   print B.l # access to a Tensor
   print B.x # access to a loop Var
   print B.U.li # multi-level access to a Tensor
   print B.U.axis[0] # multi-level access to a loop Var
   ```

<p align="right"><a href="#top">↥</a></p>
   
## 3. <a name="op">More Algorithm Description APIs</a>

In addition to `hcl.compute` and `hcl.block`, HeteroCL provides more APIs to describe an algorithm. The first one is `hcl.update`, which allows users to in-place update a Tensor.
      
   ```python
   A = hcl.update(target_tensor, input_stages, fupdate)
      
   # target[indices] = fupdate(indices)
   ```

The target tensor must appear in the list of input stages. The idea behind it is similar to `hcl.compute`, where we update each element in `target_tensor` according to `fupdate`. Similar to `hcl.block`, since we do not return any new tensor, the returned object is just a Stage.
   
HeteroCL provides a powerful API called `hcl.mut_compute`. Users can use this API to vectorize any loops.
      
   ```python
   A = hcl.mut_compute(domain, input_stages, fcompute)
   ```

In this API, `fcompute` will be executed for each index in the given `domain`. Following is an [example](/heterocl/samples/tutorial/example_3.py).
   
   ```python
   def shift_op(A, x, k):
     with hcl.if_(k == 9):
       A[k] = 0
     with hcl.else_():
       A[k] = A[k+1]
        
   A = hcl.placeholder((10,))
   B = hcl.mut_compute((5, 10), [A], lambda x, k: shift_reg(A, x, k))
      
   # equivalent code
   for x in range(5):
     for k in range(10):
       if k == 9:
         A[k] = 0
       else:
         A[k] = A[k+1]
   ```

In the above example, we shift tensor `A` to the left for five times. In each time, the right-most element becomes zero. Here we can also see the power of imperative DSL.
<p align="right"><a href="#top">↥</a></p>

## 4. <a name="badt">Bit-Accurate Data Types</a>

Most accelerators, such as FPGA, exploit bit-accurate data types for optimization. Based on this, HeteroCL provides bit-accurate data type support. To use them, you can specify the `dtype` field in most HeteroCL APIs. For the list of choices, please refer [here](#dt_table).

Another way to specify the data type is by using `hcl.downsize` or `hcl.quantize`. The main advantage of using these APIs is that we decouple the algorithm specification with the quantization scheme. Namely, the algorithm itself is independent of the quantization scheme and should have the maximum precision. The syntaxes for `hcl.downsize` and `hcl.quantize` are exactly the same except that **`hcl.downsize` is for integers while `hcl.quantize` is for floating-point numbers and fixed-point numbers**.

   ```python
   hcl.downsize(variables, dtype)
   hcl.quantize(variables, dtype)
   ```
> Note that both APIs must be used before `hcl.create_schedule`.

With these APIs, users can explore different quantization scheme using a simple loop. Following shows an [example](/heterocl/samples/tutorial/example_4.py).

   ```python
   hcl.config.init_dtype = hcl.UInt() # set the max precision
   shape = (20,)
   
   def top(dtype):
     A = hcl.placeholder(shape)
     B = hcl.compute(shape, [A], lambda x: popcount(A[x]))
     
     hcl.downsize(B, dtype) # apply different quantization schemes
     
     s = hcl.create_schedule(B)
     return hcl.build(s, [A, B])
     
   nA = np.random.randint(1 << 32, size = shape)
   nB = np.zeros(shape)
   
   print nA
   
   # 5-bit should be enough for B since the maximum popcount result is 31
   for i in range(1, 8):
     dtype = hcl.UInt(i)
     _A = hcl.asarray(nA, hcl.UInt())
     _B = hcl.asarray(nB, dtype)
     top(dtype)(_A, _B)
     print str(dtype) + ": " + str(_B)
   ```

For a more practical example, you can check the [CORDIC example](/heterocl/samples/cordic.py) in our tutorial. With bit-accurate data types, we can also perform bit operations. HeteroCL provides getting a bit, setting a bit, getting a slice, and setting a slice. For the syntax, please refer [here](#dsl).

<p align="right"><a href="#top">↥</a></p>

## 5. <a name="func">Function Block</a>

HeteroCL provides an API for users to define customized functions which can be called and scheduled. This is different from using Python functions in an imperative code block, where all the Python functions will be inlined into the code block. This API is best used when we want to reuse a logic block in several places. Namely, we do not want to instantiate the logic block multiple times on our hardware devices, which can possibly save the area. However, you can still use the `inline` scheduling function to inline a HeteroCL function block. Following shows the API.

   ```python
   f = hcl.function(input_shapes, fbody, ret_void, input_dtypes, ret_dtype, name)
   ```
   
The `input_shapes` field contains a list of shapes of the input arguments. If the input is a Var, the shape is `1`. The `fbody` field is a lambda function that implements the function body. The arguments of the lambda function are the arguments of this HeteroCL function. Currently, **the arguments must be a HeteroCL variable** (i.e., a Var or a Tensor). The `ret_void` fields indicate whether the output is void or not. The default value is `True`. The `input_dtypes` and `ret_dtype` fields specify the data type for input arguments and return value (if exists), respectively. Following we use `popcount` again as an [exmaple](/heterocl/samples/tutorial/example_5.py).

   ```python
   def popcount(n):
     l = hcl.local(0)
     with hcl.for_(0, 32) as i:
       l[0] += n[i]
     return l[0]
   
   f = hcl.function([1], lambda n: popcount(n), False)
   ```
   
After the function is defined, we can call it in any HeteroCL API.

   ```python
   m = hcl.var()
   A = hcl.placeholder((10,))
   B = hcl.compute(A.shape, [A, f], lambda x: f(A[x]))
   C = hcl.compute(A.shape, [A, f], lambda x: f(A[x] & m))
   
   s = hcl.create_schedule([B, C])
   ```
   
Note that `f` is also an input stage to the two `compute` APIs. In the above example, we use `popcount` twice. The first time we calculate the number of ones while for the second time we count the number of ones after applying a mask `m`. If we use a Python function as in the previous examples, `popcount` will be inlined to both `compute` APIs. However, here we have two function calls. You can examine the generated code by printing out the IR using `hcl.lower`.

<p align="right"><a href="#top">↥</a></p>

## 6. <a name="red">Reduction Operations</a>

Reduction operations are useful in many designs. One example is summation. To write a reduction operation in HeteroCL, first, we need to define a reduction axis.

   ```python
   r = hcl.reduce_axis(min, max, name)
   ```
   
The arguments to the reduction axis are the minimum and maximum value of the reduction range and a name (optional), respectively. After we have the reduction axis, we can define our reducer.

   ```python
   def freduce(val, acc):
     # compute and return new_acc or update acc
   
   R = hcl.reducer(init_var, freduce, dtype)
   ```
   
The reducer takes in an initiation variable, which can be either a Var or a Tensor. This is also our initiation accumulator. The second argument is the reduction function, which takes in a value and an accumulator. **The order and the number of the arguments must not change.**

> If the accumulator is not a Tensor, you need to return the new accumulator. Otherwise, you can update it. 

The last argument specifies the data type of both the value and the accumulator. Following shows an example of how to define a summation operation.

   ```python
   sum = hcl.reducer(0, lambda x, y: x + y)
   ```

Now we have both the reduction axis and the reducer. We can write the reduction operation by combining both.

   ```python
   ret = R(input_tensor, axis = list_of_axis, where = cond)
   ```

The shape of the returned value is the same as the accumulator. Finally, we can combine this with the `hcl.compute` API. *Note that you can only use the reducer inside `hcl.compute` or `hcl.update`.* Following shows an [example](/heterocl/samples/tutorial/example_6_1.py) of summing up all the elements in a 2D tensor if the element is greater than zero. Here we make use of the predefined reducer `hcl.sum`.

   ```python
   k1 = hcl.reduce_axis(0, 10)
   k2 = hcl.reduce_axis(0, 10)
   
   A = hcl.placeholder((10, 10))
   B = hcl.compute((1,), [A], 
             lambda x: hcl.sum(A[k1, k2], axis = [k1, k2], where = A[k1, k2] > 0))
   ```

We can also write something more complicated with customized reducer. In the following [example](/heterocl/samples/tutorial/example_6_2.py), we find the largest two numbers for each row in a 2D tensor.

   ```python
   def find_max2(val, acc):
     with hcl.if_(val > acc[0]):
       with hcl.if_(val < acc[1]):
         acc[0] = val
       with hcl.else_():
         acc[0] = acc[1]
         acc[1] = val
   
   k = hcl.reduce_axis(0, 10)
   init = hcl.compute((2,), [], lambda x: 0)
   R = hcl.reducer(init, find_max2)
   
   A = hcl.placeholder((10, 10))
   B = hcl.compute((2, 10), [A, init], lambda _, y: R(A[k, y], axis = k))
   ```

In the above example, we do not return any new accumulator in `find_max2` because `acc` is a Tensor. Also, in `hcl.compute` API, we need to include the `init` Tensor since it is also an input stage to `B`. Note that for our lambda function, we use `_` to specify the axis being reduced. This also satisfy the requirement that the length of the output shape matches the number of lambda arguments.

<p align="right"><a href="#top">↥</a></p>
