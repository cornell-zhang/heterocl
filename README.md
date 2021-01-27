[![GitHub license](https://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![CircleCI](https://circleci.com/gh/cornell-zhang/heterocl/tree/master.svg?style=svg&circle-token=2b5ee9faf30b94aac41b61032d03e4654a65079d)](https://circleci.com/gh/cornell-zhang/heterocl/tree/master)

HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing
===================================================================================================

[Website](http://heterocl.csl.cornell.edu/web/index.html) | [Installation](http://heterocl.csl.cornell.edu/doc/installation.html) | [Tutorials](http://heterocl.csl.cornell.edu/doc/tutorials/index.html) | [Samples](http://heterocl.csl.cornell.edu/doc/samples/index.html) | [Documentation](http://heterocl.csl.cornell.edu/doc/index.html)

## Introduction

With the pursuit of improving compute performance under strict power constraints, there is an increasing need for deploying applications to heterogeneous hardware architectures with accelerators, such as GPUs and FPGAs. However, although these heterogeneous computing platforms are becoming widely available, they are very difficult to program especially with FPGAs. As a result, the use of such platforms has been limited to a small subset of programmers with specialized hardware knowledge.

To tackle this challenge, we introduce HeteroCL, a programming infrastructure comprised of a Python-based domain-specific language (DSL) and a compilation flow. The HeteroCL DSL provides a clean programming abstraction that decouples algorithm specification from three important types of hardware customization in compute, data types, and memory architectures. HeteroCL can further capture the interdependence among these different customization techniques, allowing programmers to explore various performance/area/accuracy trade-offs in a systematic and productive manner. In addition, our framework currently provides two advanced domain-specific optimizations with stencil analysis and systolic array generation, which produce highly efficient microarchitectures for accelerating popular workloads from image processing and deep learning domains.

## Current Compilation Flow

![flow](docs/github-flow.png)

## Evaluation on AWS F1 (Xilinx Virtex UltraScale+<sup>TM</sup> VU9P FPGA)
The speedup is over a single-core single-thread CPU execution on AWS F1.

| Benchmark | Data Sizes & Type | #LUTs | #FFs | #BRAMs | #DSPs | Freqency (MHz) | Speedup | Back End |
| :-------- | :---------------- | :----: | :----:| :-----: | :----: | :------------: | :------:| :------: |
| **KNN Digit Recognition**<br/>Image classification | K=3 #images=1800<br/>`uint49` | 4009 | 5835 | 88 | 0 | 250 | 12.5 | General |
| **K-Means**<br/> Clustering | K=16 #elem=320 x 32<br/>`int32` | 212708 | 235011 | 32 | 1536 | 190.6 | 16.0 | General |
| **Smith-Waterman**<br/> Genomic sequencing | string len=128<br/>`uint2` | 110841 | 88369 | 1409 | 0 | 152.2 | 20.9 | General |
| **Seidel**<br/> Image processing | 2160 pixel x 3840 pixel<br/>`fixed16` | 21719 | 31663 | 46 | 96 | 250 | 5.9 | Stencil |
| **Gaussian**<br/> Image processing | 2160 pixel x 3840 pixel<br/>`fixed16` | 70833 | 131160 | 46 | 688 | 250 | 13.2 | Stencil |
| **Jacobi**<br/> Linear algebra | 2160 pixel x 3840 pixel<br/>`fixed16` | 14883 | 22485 | 46 | 48 | 250 | 5.0 | Stencil |
| **GEMM**<br/> Matrix multiplication | 1024 x 1024 x 1024<br/>`fixed16` | 454492 | 800283 | 932 | 2507 | 236.8 | 8.9 | Systolic Array |
| **LeNet Inference**<br/> CNN | MNIST<br/>`fixed16` | 362291 | 660186 | 739.5 | 1368 | 250 | 10.6 | Systolic Array |


## Publication

If you use HeteroCL in your design, please cite our [FPGA'19 paper](http://www.csl.cornell.edu/~zhiruz/pdfs/heterocl-fpga2019.pdf):
```
@article{lai2019heterocl,
  title={HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing},
  author={Lai, Yi-Hsiang and Chi, Yuze and Hu, Yuwei and Wang, Jie and Yu, Cody Hao and 
          Zhou, Yuan and Cong, Jason and Zhang, Zhiru},
  journal={Int'l Symp. on Field-Programmable Gate Arrays (FPGA)},
  year={2019}
}
```

## Related Work

HeteroCL is a Python-based DSL extended from TVM and it extends Halide IR for intermediate representation. HeterCL incoporates the SODA framework, PolySA framework, and Merlin Compiler for FPGA back-end generation.

* **[Stencil with Optimized Dataflow Architecture](https://vast.cs.ucla.edu/~chiyuze/pub/iccad18.pdf)** (SODA)
* **[Polyhedral-Based Systolic Array Auto-Compilation](http://cadlab.cs.ucla.edu/~jaywang/papers/iccad18-polysa.pdf)** (PolySA)
* **[Merlin Compiler](https://www.falconcomputing.com/merlin-fpga-compiler/)**
* **[Halide](https://halide-lang.org)**
* **[TVM](https://tvm.ai)**

## Contributing to HeteroCL

### Coding Style (Python)

We follow [official Python coding style](https://www.python.org/dev/peps/pep-0008/#descriptive-naming-styles) and use [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html#other-points-to-keep-in-mind).

### Coding Style (C and C++)

We follow [Google coding style](https://google.github.io/styleguide/cppguide.htm).

### Steps

1. Use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to format your C-related files. The configuration file is in `docs/.clang-format`. Following is a sample command to format the file in place. Note that you need to put the configuration file at the same directory you execute the command.

   ``clang-format -i -style=file <cpp-file>``
2. Use [Pull Request](https://help.github.com/articles/about-pull-requests/). Remember to select the most suitable labels and put it in the title.
3. Make sure all the tests pass.
