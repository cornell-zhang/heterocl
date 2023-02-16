import subprocess
import setuptools
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
    """Custom build command."""

    def run(self):
        build_py.run(self)
        cmd = "cd scripts && ./build_llvm.sh"
        subprocess.Popen(cmd, shell=True).wait()


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
        setup_requires=[],
        cmdclass={
            "build_py": BuildPyCommand,
        },
        install_requires=parse_requirements("requirements.txt"),
        packages=setuptools.find_packages(),
        url="https://github.com/cornell-zhang/heterocl",
        python_requires=">=3.6",
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Accelerator Design",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )


if __name__ == "__main__":
    setup()
