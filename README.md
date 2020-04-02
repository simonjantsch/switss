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
    * pulp

## Sphinx (Documentation)
* install Sphinx via `pip3 install sphinx`
* generate documentation by doing the following steps: 
```sh
cd docs
make html
xdg-open build/html/index.html
```
    
## Solvers
By installing `PuLP`, the CBC-solver is automatically installed alongside of it. In order to use the Gurobi solver, 
`PuLP` needs to be configured (cf. [here](https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html)) 
by adding the following lines to the .bash_profile:

    export GUROBI_HOME="[gurobi install path]/linux64"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export LD_LIBRARY_PATH="${GUROBI_HOME}/lib"

and when Windows is used, call

    set GUROBI_HOME=[gurobi install path]/linux64
    set PATH=%PATH%;%GUROBI_HOME%/bin
    set LD_LIBRARY_PATH=%LD_LIBRARY_PATH%;%GUROBI_HOME%/lib

via command line or graphical user interface.