<a name="top"></a>

<p align="right"><a href="README.md">Back to API</a></p>

| API | Explanation |
| :-- | :-- |
| [```hcl.var([name, dtype])```](#var) | Create a variable with specified name and data type |
| [```hcl.placeholder(shape[, name, dtype])```](#ph) | Create a placeholder with specified shape, name, and data type |
| [```hcl.compute(shape, fcompute[, name, inline])```](#com) | A computation function that executes fcompute and returns a new tensor |
| [```hcl.update(target, fcompute[, name, inline])```](#upd) | An update function that executes fcompute and updates target |
| [```hcl.block(func, *args)```](#block) | Create a stage for an imperative code block |

***

#### <a name="var">```hcl.var(name = "var", dtype = "int32")```</a> 
Create a variable with the specified name and data type. The default datatype is int32.

Parameters:
* name (`str`, optional): the name
* dtype (`str`, optional) : the data type

Return type: `Var`

Example:
```python
a = hcl.var(name = "a", dtype = "float32")
```

<p align="right"><a href="#top">↥</a></p>

***

#### <a name="ph">```hcl.placeholder(shape, name = "placeholder", dtype = "int32")```</a>
Create a placeholder with the specified shape, name, and data type. The default datatype is int32.

Parameters:
* shape (`tuple`): a tuple of integers
* name (`str`, optional): the name
* dtype (`str`, optional) : the data type

Return type: [`Tensor`](tensor.md#tensor)

Example:
```python
a = hcl.placeholder((10,), name = "a", dtype = "float32") # a 1D placeholder
a = hcl.placeholder((10, 10), name = "a", dtype = "int8") # a 2D placeholder
```
<p align="right"><a href="#top">↥</a></p>

***

#### <a name="com">```hcl.compute(shape, fcompute, name = "compute", inline = True)```</a>
A computation function that executes fcompute on the given indices and returns a new tensor.

`output[index] = fcompute(index)`

Parameters:
* shape (`tuple`): a tuple of integers
* fcompute (`lambda`): a lambda function with inputs as indices of the tensors and body that specifies the compute rule.
* name (`str`, optional): the name
* inline (`bool`, optional): whether fcompute should be inlined or not. The default value is `True`.

Return type: `Tensor`

Example 1:
```python
C = hcl.compute((10, 10), lambda x, y: A[x, y] * B[x, y])
# the above line is equvilant to
for x in range(0, 10):
  for y in range(0, 10):
    C[x][y] = A[x][y] * B[x][y]
```
Example 2:
```python
def myfun(a):
  b = 0
  for i in range(0, 3):
    b = b + a
  return b

B = hcl.compute((10,), lambda x: myfun(A[x]), inline = True)
# the above line is equivalent to
for x in range(0, 10):
  B[x] = A[x] + A[x] + A[x]

B = hcl.map((10,), lambda x: myfunc(A[x]), inline = False)
# the above line is equivalent to
for x in range(0, 10):
  B[x] = 0
  for i in range(0, 3):
    B[x] = B[x] + A[x]
```
<p align="right"><a href="#top">↥</a></p>

***

#### <a name="upd">```hcl.update(target, fcompute, name = "update", inline = True)```</a>
An update function that executes fcompute on the given indices and updates the target tensor.

Parameters:
* target (`Tensor`): the target to be updated
* fcompute (`lambda`): a lambda function with inputs as indices of the tensors and body that specifies the compute rule.
* name (`str`, optional): the name
* inline (`bool`, optional): whether fcompute should be inlined or not. The default value is `True`.

Return type: `Stage`

Example:
```python
A = tvm.placeholder((10, 10), name = "A")
tvm.update(A, lambda x, y: A[x, y] + 1)
# the above line is equivalent to
for x in range(0, 10):
  for y in range(0, 10):
    A[x][y] = A[x][y] + 1
```
<p align="right"><a href="#top">↥</a></p>

***

#### <a name="block">```hcl.block(func, *args)```</a>
Create an imperative code block and return the stage.

Parameters:
* func (`function`): the function that contains imperative code
* args (`list`): a list of arguments, could be `Var` or `Tensor`

Return type: `Stage`

Example:
```python
def imp_code(A, B):
  if A[0] == 2:
    B[0] = 3
    
A = tvm.placeholder((10,), name = "A")
B = tvm.placeholder((5,), name = "B")
tvm.block(imp_code, [A, B])
# the above line is equivalent to
tvm.update(B, lambda x: tvm.select(x == 0, tvm.select(A[0] == 2, 3, B[x]), B[x]))
```
<p align="right"><a href="#top">↥</a></p>
