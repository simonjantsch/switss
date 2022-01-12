import numpy
import sys
from setuptools import setup, find_packages
from Cython.Build import cythonize

install_packages = [ "graphviz",
                     "scipy",
                     "numpy",
                     "bidict",
                     "pulp",
                     "cython" ]

cython_modules = [ "switss/utils/graph.pyx",
                   "switss/utils/treaps.pyx",
                   "switss/utils/stack.pyx"]

if "--with-treealg" in sys.argv:
    install_packages += ["Pillow"]
    cython_modules += ["switss/utils/tree_decomp.pyx"]
    sys.argv.remove("--with-treealg")


setup(
    name = 'switss',
    version = '0.1',
    packages = find_packages(),
    scripts=['bin/switss'],
    ext_modules = cythonize(cython_modules),
    include_dirs=[numpy.get_include()],
    install_requires = install_packages
)
