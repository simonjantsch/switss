from setuptools import Extension

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name = 'switss',
    version = '0.1',
    packages = find_packages(),
    scripts=['bin/switss'],
    ext_modules = cythonize("switss/utils/graph.pyx") + cythonize("switss/utils/treaps.pyx") + cythonize("switss/utils/stack.pyx"),
    install_requires=[
        "graphviz",
        "scipy",
        "numpy",
        "bidict",
        "pulp",
        "cython"
    ]
)
