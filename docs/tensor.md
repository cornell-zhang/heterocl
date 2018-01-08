<a name="top"></a>

<p align="right"><a href="README.md">Back to API</a></p>

| API | Effect |
| :-- | :-- |
| [```hcl.tensor.Tensor```](#tensor) | The base class for HeteroCL tensors |

***

#### <a name="tensor">```class hcl.tensor.Tensor```</a> 
The base class for HeteroCL tensors.

Bases: [`tvm.tensor.Tensor`](http://docs.tvmlang.org/api/python/tensor.html#tvm.tensor.Tensor)

| Method | Effect |
| :-- | :-- |
| [`ndim`](#ndim) | Number of dimensions |
| [`stages`](#stages) | Get the stages (including definition and updates) of the Tensor |


##### <a name="ndim">```ndim```</a>
The number of dimensions

Type: `Integer`

<p align="right"><a href="#top">↥</a></p>

##### <a name="stages">```stages```</a>
Get the stages (including definition and updates) of the Tensor. The first one (i.e., `stages[0]`) returns the definition.

Type: `list` of `Stage`

<p align="right"><a href="#top">↥</a></p>
