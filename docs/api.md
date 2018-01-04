| API | Explanation |
| :-- | :-- |
| [```hcl.var([name, dtype])```](#var) | Create a variable with specified name and data type |
| [```hcl.placeholder(shape[, name, dtype])```](#ph) | Create a placeholder with specified shape, name, and data type |
| [```hcl.compute(shape, fcompute[, name])```](#com) | A map function that executes fcompute, where fcompute is inlined |
| [```hcl.map(shape, fcompute[, name])```](#map) | A map function that executes fcompute, where fcompute will not be inlined |

***

#### <a name="var">```hcl.var(name = "var", dtype = "int32")```</a>
Create a variable with the specified name and data type. The default datatype is int32.

Parameters:
* name (`str`, optional): the name
* dtype (`str`, optional) : the data type

Return type: `Var`

```python
a = hcl.var(name = "a", dtype = "float32")
```

***

#### <a name="ph">```hcl.placeholder(shape, name = "placeholder", dtype = "int32")```</a>
Create a placeholder with the specified shape, name, and data type. The default datatype is int32.

Parameters:
* shape (`tuple`): a tuple of integers
* name (`str`, optional): the name
* dtype (`str`, optional) : the data type

Return type: `Tensor`

```python
a = hcl.placeholder((10,), name = "a", dtype = "float32") # a 1D placeholder
a = hcl.placeholder((10, 10), name = "a", dtype = "int8") # a 2D placeholder
```

***

#### <a name="com">```hcl.compute(shape, fcompute, name = "compute")```</a>
A map function that executes fcompute on each element of input tensors and returns a new tensor.

Parameters:
* shape (`tuple`): a tuple of integers
* fcompute (`lambda`): a lambda function with inputs as indices of the tensors and body as the function applied to each tensor element. See the examples for more details. Everything inside the body will be automatically inlined. To avoid this, use `hcl.map` instead.
* name (`str`, optional): the name

Return type: `Tensor`

```python
C = hcl.compute((10, 10), lambda x, y: A[x, y] * B[x, y])
# the above line is equvilant to
for x in range(0, 10):
  for y in range(0, 10):
    C[x][y] = A[x][y] * B[x][y]
```

***

#### <a name="map">```hcl.map(shape, fcompute, name = "map")```</a>
Similar to `hcl.compute` except that the body inside fcompute will not be inlined. This is useful if we want to schedule fcompute.

Parameters:
* shape (`tuple`): a tuple of integers
* fcompute (`lambda`): a lambda function with inputs as indices of the tensors and body as the function applied to each tensor element.
* name (`str`, optional): the name

Return type: `Tensor`

```python
def myfun(a):
  b = 0
  for i in range(0, 3):
    b = b + a
  return b

B = hcl.compute((10,), lambda x: myfun(A[x]))
# the above line is equvilant to
for x in range(0, 10):
  for y in range(0, 10):
    B[x] = A[x] + A[x] + A[x]

B = hcl.map((10,), lambda x: myfunc(A[x]))
# the above line is equvilant to
for x in range(0, 10):
  for y in range(0, 10):
    B[x] = 0
    for i in range(0, 3):
      B[x] = B[x] + A[x]
```
