import setuptools


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
        install_requires=[
            "numpy",
            "xmltodict",
            "tabulate",
            "pytest",
            "networkx",
            "matplotlib",
            "pandas",
            "imageio",
            "psutil",
            "sympy",
        ],
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
