[![GitHub license](https://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![CircleCI](https://circleci.com/gh/cornell-zhang/heterocl/tree/master.svg?style=svg&circle-token=2b5ee9faf30b94aac41b61032d03e4654a65079d)](https://circleci.com/gh/cornell-zhang/heterocl/tree/master)

HeteroCL: Heterogenous Computing Language
=========================================

[Documentation](http://heterocl.csl.cornell.edu/docs/build/html/index.html)

## Introduction

With the pursuit of improving compute performance under strict power constraints, there is an increasing need for deploying applications to heterogeneous hardware architectures with accelerators, such as GPUs and FPGAs. However, although these heterogeneous computing platforms are becoming widely available, they are very difficult to program especially with FPGAs. As a result, the use of such platforms has been limited to a small subset of programmers with specialized hardware knowledge.

To tackle this challenge, we introduce HeteroCL, a programming infrastructure comprised of a Python-based domain-specific language (DSL) and an FPGA-targeted compilation flow. The HeteroCL DSL provides a clean programming abstraction that decouples algorithm specification from three important types of hardware customization in compute, data types, and memory architectures. HeteroCL can further capture the interdependence among these different customization techniques, allowing programmers to explore various performance/area/accuracy trade-offs in a systematic and productive manner. In addition, our framework currently provides two advanced domain-specific optimizations with stencil analysis and systolic array generation, which produce highly efficient microarchitectures for accelerating popular workloads from image processing and deep learning domains.

## Publication

If you use HeteroCL in your design, please cite our [FPGA'19 paper](http://www.csl.cornell.edu/~zhiruz/pdfs/heterocl-fpga2019.pdf):
```
@article{lai2019heterocl,
  title={HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing},
  author={Lai, Yi-Hsiang and Chi, Yuze and Hu, Yuwei and Wang, Jie and Yu, Cody Hao and 
          Zhou, Yuan and Cong, Jason and Zhang, Zhiru},
  journal = {Int'l Symp. on Field-Programmable Gate Arrays (FPGA)},
  year={2019}
}
```

## Contributing to HeteroCL
1. Use [Pull Request](https://help.github.com/articles/about-pull-requests/).
2. Python [coding style](https://www.python.org/dev/peps/pep-0008/#descriptive-naming-styles).
3. Python [docstring style](https://numpydoc.readthedocs.io/en/latest/format.html#other-points-to-keep-in-mind).
