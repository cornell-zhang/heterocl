## FPGA Backend Development

### Create a new FPGA backend (e.g., "hls")

#### Part 1: Create code generator
* Create `tvm/src/codegen/codegen_hls.cc`
* Create `tvm/src/codegen/codegen_hls.h`
* Create `tvm/src/codegen/build_hls.cc`

Note: `TVM_REGISTER_API` is used to bind the C++ function to Python,
so `TVM_REGISTER_API("codegen.build_hls")` means you can now call
`codegen.build_hls` in the TVM Python side.

**IMPORTANT**: You **must** name this API as `codegen.build_target` where
`target` is the name you wish user to specify.
		
#### Part 2: Register the code generator to the TVM Python interface
* In `tvm/python/target.py`:
    * Find the list named `FPGA_TARGETS`
    * Add your backend language name to the list. For example:
		```python
		FPGA_TARGETS = ['hls']
		```
#### Part 3: Testing
Use the HeteroCL API to assign the target:
```python
f = hcl.build(s, [a, A, B], target='hls')
```
