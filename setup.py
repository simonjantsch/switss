from setuptools import setup, find_packages
setup(
    name = 'farkas',
    version = '0.1',
    packages = ["farkas"],
    install_requires=[
        "graphviz",
        "scipy",
        "numpy",
        "bidict",
        "pulp",
    ]
)