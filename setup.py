import numpy
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name = 'switss',
    version = '0.1',
    packages = find_packages(),
    scripts=['bin/switss'],
    ext_modules = cythonize(["switss/utils/graph.pyx","switss/utils/tree_decomp.pyx"]),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "graphviz",
        "scipy",
        "numpy",
        "bidict",
        "pulp",
    ]
)
