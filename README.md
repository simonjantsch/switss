## Requirements
* **system** (via `apt-get install`)
    * python3 (>=3.7)
    * graphviz
    * [prism](https://www.prismmodelchecker.org/download.php). The `bin/` directory of prism must be added to the `$PATH` variable.

* **python** (via `pip3 install`)
    * graphviz
    * scipy
    * numpy
    * bidict

## Sphinx (Documentation)
* install Sphinx via `pip3 install sphinx`
* generate documentation by doing the following steps: 
```sh
cd docs
make html
xdg-open build/html/index.html
```
