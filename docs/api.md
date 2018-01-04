| API | Explanation |
| :-- | :-- |
| [```hcl.var([name, dtype])```](#var) | Create a variable with specified name and data type |
| [```hcl.placeholder(shape[, name, dtype])```](#ph) | Create a placeholder with specified shape, name, and data type |
| [```hcl.compute(shape, fcompute[, name])```](#com) | A map function that executes fcompute |

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
A map function that executes fcompute on each input tensors and returns a new tensor.

Parameters:
* shape (`tuple`): a tuple of integers
* fcompute (`lambda`): a lambda function with inputs as indices of the tensors and body as the function applied to each tensor element. See the examples for more details.
* name (`str`, optional): the name

Return type: `Tensor`

```python
C = hcl.compute((10, 10), lambda x, y: A[x, y] * B[x, y])
# the above line is equvilant to
for x in range(0, 10):
  for y in range(0, 10):
    C[x][y] = A[x][y] * B[x][y]
```
