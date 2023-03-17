# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
import setuptools
import distutils
from distutils.command.build import build as _build
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

args = sys.argv[1:]
nthreads = args[args.index("--nthreads") + 1] if "--nthreads" in args else 4


class CustomBuild(_build):
    def finalize_options(self):
        # call parent finalize_options method
        _build.finalize_options(self)

        self.src_dir = os.path.abspath(os.path.dirname(__file__))
        self.llvm_dir = os.path.join(self.src_dir, "hcl-dialect/externals/llvm-project")
        if not os.path.exists(os.path.join(self.llvm_dir, "llvm")):
            raise RuntimeError(
                "`llvm-project` not found. Please run `git submodule update --init --recursive` first"
            )

    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class CMakeBuild(build_py):
    """Custom build command."""

    def initialize_options(self):
        # call parent initialize_options method
        build_py.initialize_options(self)

    def finalize_options(self):
        # call parent finalize_options method
        build_py.finalize_options(self)

    def build_llvm(self):
        self.announce(
            f"Building LLVM 15.0.0 in {self.llvm_cmake_build_dir}...",
            level=distutils.log.INFO,
        )
        llvm_cmake_args = [
            "-DLLVM_ENABLE_PROJECTS=mlir",
            "-DLLVM_BUILD_EXAMPLES=ON",
            "-DLLVM_TARGETS_TO_BUILD=host",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            "-DLLVM_INSTALL_UTILS=ON",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            "-DPython3_EXECUTABLE=`which python3`",
        ]
        os.makedirs(self.llvm_cmake_build_dir, exist_ok=True)
        result = subprocess.run(
            ["cmake", "-G", "Unix Makefiles", self.llvm_dir + "/llvm"]
            + llvm_cmake_args,
            cwd=self.llvm_cmake_build_dir,
            env=os.environ.copy(),
            check=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "LLVM CMake failed. Please check the error message above."
            )
        result = subprocess.run(
            ["make", f"-j{nthreads}"], cwd=self.llvm_cmake_build_dir
        )
        if result.returncode != 0:
            raise RuntimeError(
                "LLVM make failed. Please check the error message above."
            )

    def build_hcl_dialect(self):
        self.announce(
            f"Building HeteroCL Dialect in {self.hcl_cmake_build_dir}...",
            level=distutils.log.INFO,
        )
        hcl_cmake_args = [
            f"-DMLIR_DIR={self.llvm_cmake_build_dir}/lib/cmake/mlir",
            f"-DLLVM_EXTERNAL_LIT={self.llvm_cmake_build_dir}/bin/llvm-lit",
            "-DPYTHON_BINDING=ON",
            "-DOPENSCOP=OFF",
            "-DPython3_EXECUTABLE=`which python3`",
        ]
        os.makedirs(self.hcl_cmake_build_dir, exist_ok=True)
        result = subprocess.run(
            ["cmake", "-G", "Unix Makefiles", self.hcl_dialect_dir] + hcl_cmake_args,
            cwd=self.hcl_cmake_build_dir,
            env=os.environ.copy(),
            check=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "HCL-MLIR CMake failed. Please check the error message above."
            )
        result = subprocess.run(["make", f"-j{nthreads}"], cwd=self.hcl_cmake_build_dir)
        if result.returncode != 0:
            raise RuntimeError(
                "HCL-MLIR make failed. Please check the error message above."
            )

    def install_hcl_dialect(self):
        self.announce("Installing hcl_mlir python package...", level=distutils.log.INFO)
        result = subprocess.run(
            ["pip", "install", "-e", "tools/hcl/python_packages/hcl_core"],
            cwd=self.hcl_cmake_build_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "HCL-MLIR Python package installation failed. Please check the error message above."
            )

    def run(self):
        self.src_dir = os.path.abspath(os.path.dirname(__file__))
        self.llvm_dir = os.path.join(self.src_dir, "hcl-dialect/externals/llvm-project")
        if not os.path.exists(os.path.join(self.llvm_dir, "llvm")):
            raise RuntimeError(
                "`llvm-project` not found. Please run `git submodule update --init --recursive` first"
            )
        self.llvm_cmake_build_dir = os.path.join(self.llvm_dir, "build")
        if not os.path.exists(
            os.path.join(self.llvm_cmake_build_dir, "bin/llvm-config")
        ):
            self.build_llvm()
        self.hcl_dialect_dir = os.path.join(self.src_dir, "hcl-dialect")
        self.hcl_cmake_build_dir = os.path.join(self.hcl_dialect_dir, "build")
        if not os.path.exists(
            os.path.join(self.src_dir, "hcl-dialect/build/bin/hcl-opt")
        ):
            self.build_hcl_dialect()
            self.install_hcl_dialect()
        build_py.run(self)


class NoopBuildExtension(build_ext):
    def build_extension(self, ext):
        pass


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


def setup():
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setuptools.setup(
        name="heterocl",
        description="HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing",
        version="0.5",
        author="HeteroCL",
        long_description=long_description,
        long_description_content_type="text/markdown",
        cmdclass={
            "build": CustomBuild,
            "build_py": CMakeBuild,
            "build_ext": NoopBuildExtension,
        },
        setup_requires=["numpy", "pybind11", "pip", "cmake"],
        install_requires=parse_requirements("requirements.txt"),
        packages=setuptools.find_packages(),
        url="https://github.com/cornell-zhang/heterocl",
        python_requires=">=3.7",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: System :: Hardware",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )


if __name__ == "__main__":
    setup()
