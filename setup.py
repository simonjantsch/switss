from setuptools import setup, find_packages

setup(
    name = 'switss',
    version = '0.1',
    packages = find_packages(),
    scripts=['bin/switss'],
    install_requires=[
        "graphviz",
        "scipy",
        "numpy",
        "bidict",
        "pulp",
    ]
)
