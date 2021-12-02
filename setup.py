from setuptools import Extension

from setuptools import setup, find_packages
from Cython.Build import cythonize

ext_modules = [Extension("switss/utils/graph", ["switss/utils/graph.pyx"])]
for e in ext_modules:
    e.cython_directives = {"embedsignature": True}
    
setup(
    name = 'switss',
    version = '0.1',
    packages = find_packages(),
    scripts=['bin/switss'],
    ext_modules = cythonize("switss/utils/graph.pyx") + ext_modules,
    install_requires=[
        "graphviz",
        "scipy",
        "numpy",
        "bidict",
        "pulp",
        "cython"
    ]
)
