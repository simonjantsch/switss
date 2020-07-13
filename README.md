# SWITSS
A tool for the computation of Small WITnessing SubSystems in Markov Decision Processes (MDPs) and Discrete Time Markov Chains (DTMCs).
SWITSS implements exact and heuristic methods for computing small witnessing subsystems by reducing the problem to (mixed integer)
linear programming. Returned subsystems can automatically be rendered graphically and are accompanied with a certificate which proves 
that the subsystem is indeed a witness.

See https://simonjantsch.github.io/switss-docs/ for the documentation.

## Requirements
* python3
* graphviz
* [prism](https://www.prismmodelchecker.org/download.php). The `bin/` directory of prism must be added to the `$PATH` variable,
i.e. add `export PATH="[prism install path]/bin:$PATH"` to your `~/.bashsrc`.

## Installation
Run `sudo python3 setup.py install`.
    
## Solvers
By installing `PuLP`, the CBC-solver is automatically installed alongside of it. In order to use the Gurobi solver, 
`PuLP` needs to be configured (cf. [here](https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html)) 
by adding the following lines to `~/.bashrc`:

    export GUROBI_HOME="[gurobi install path]/linux64"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export LD_LIBRARY_PATH="${GUROBI_HOME}/lib"
    
## Tests
After installing, call `pytest --pyargs switss -v` in order to run all tests. In `switss/test/test_dtmc.py` and `switss/test/test_mdp.py` variables `lp_solver` and `milp_solvers` specify lists of used solvers. If one of these solvers happens to be not installed on your system, you can just delete it from the lists and re-install switss.

## SWITSS Command Line Tool
SWITSS comes with a handy command line tool that makes it possible to find minimal witnessing subsystems without having to open up
python. It supports 

* displaying system information,
* transforming systems into a canonical "reachability form",
* minimization of reachability forms,
* validity checks of generated certificates,
* and rendering of models as .svg files.

For a detailed description on how to use this tool call `switss --help`. Here are a few examples that can be executed in the root
directory of this repository:

Displaying information about the DTMC in `examples/groups.ipynb`:

    switss info dtmc examples/datasets/groups-example

Rendering that DTMC:

    switss render dtmc examples/datasets/groups-example -vi

Creating & storing a reachability form from that model:

    switss rf dtmc examples/datasets/groups-example
    
Minimizing said reachability form by using the quotient sum heuristic on the "min"-form:

    switss minimize dtmc examples/datasets/groups-example-rf -m QSHeur mode=min -s threshold=0.5

Check validity of generated certicates:

    switss certify dtmc examples/datasets/groups-example-rf -c examples/datasets/groups-example-rf.certificates.json

Generate subsystem based on certificate:

    switss subsystem dtmc examples/datasets/groups-example-rf -c examples/datasets/groups-example-rf.certificates.json

Render generated subsystem:

    switss render dtmc examples/datasets/groups-example-rf-subsys-0 -vi -col rf_target,rf_fail,subsystem

Minimize again, but this time based on labels & with another method:

    switss minimize dtmc examples/datasets/groups-example-rf -m MILPExact mode=min -s threshold=0.5 labels=group1,group2,group3
