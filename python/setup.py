from setuptools import setup

setup(
  name = "heterocl",
  version = "1.0.0",
  install_requires=[
      'numpy==1.18.5',
      'decorator==4.4.2',
      'networkx==2.5.1',
      'matplotlib',
      'backports.functools_lru_cache',
      'ordered_set',
      'xmltodict',
      'tabulate',
      'sodac',
    #   'pandas', # timed out on CI
      ])
