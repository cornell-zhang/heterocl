## HeteroCL Documentation


### Installation Guide

#### System Requirments

cmake >= 3.4.3

#### Steps
1. Clone the repo with `--recursive` option to make sure the submodules are also cloned.
2. HeteroCL requires LLVM >= 4.0. If your version meets the requirement, edit `Makefile.config` and point the path of `llvm-config` to `LLVM_CONFIG`. Otherwise, the next step will help you install LLVM with the required version.
3. Build the whole project by `make`.
4. Update your Python path
```bash
export PYTHONPATH=/path/to/tvm/python:/path/to/hcl/python:$PYTHONPATH
```

### Frequently Asked Questions
### Python API
* [hcl](api.md)
* [hcl.tensor](tensor.md)
* hcl.schedule
