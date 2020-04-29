from setuptools import setup, find_packages

setup(
    name = 'farkas',
    version = '0.1',
    packages = find_packages(),
    install_requires=[
        "graphviz",
        "scipy",
        "numpy",
        "bidict",
        "pulp",
    ]
)