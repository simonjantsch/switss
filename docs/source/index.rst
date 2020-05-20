.. switss documentation master file, created by
   sphinx-quickstart on Mon Mar 16 15:04:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######
SWITSS
######

A tool for the computation of Small WITnessing SubSystems in Markov Decision Processes (MDPs) and 
Discrete Time Markov Chains (DTMCs). SWITSS implements exact and heuristic methods for computing small 
witnessing subsystems by reducing the problem to (mixed integer) linear programming. Returned subsystems 
can automatically be rendered graphically and are accompanied with a certificate which proves that 
the subsystem is indeed a witness. The work is based on [FJB19]_.
A precondition on the MDP that can be handled is that the probability to reach goal or fail is positive under each scheduler.

Contact: hans.harder@mailbox.tu-dresden.de

**********************
MDP and DTMC classes
**********************

A MDP is a tuple :math:`\mathcal{M} = (S_{\text{all}}, \text{Act}, \textbf{P}, s_0)`, where 

* :math:`S_{\text{all}}` denotes the set of states,
* :math:`\text{Act}` denotes the set of actions,
* :math:`s_0` is the initial state,
* and :math:`\textbf{P}: S_{\text{all}} \times \text{Act} \times S_{\text{all}} \rightarrow [0,1]` the transition probability function.
   
Also,

* :math:`\text{Act}(s) \subseteq \text{Act}` denotes the set of actions that are enabled in state :math:`s`.
* For a set of states :math:`S` and whenever suitable, :math:`\mathcal{M}_S = \{ (s,a) \in S \times \text{Act} \mid a \in \text{Act}(s) \}` also denotes the set of state-action-pairs.
* For a set of states :math:`S`,  :math:`C_{S} = | \mathcal{M}_{S} |` denotes the amount of state-action-pairs and :math:`N_S = | S |` the amount of states.

For :math:`\textbf{P}` we will use a :math:`C_{S_{\text{all}}} \times N_{S_{\text{all}}}` transition matrix. Furthermore, every 
state-action pair :math:`(s,a) \in \mathcal{M}_{S_{\text{all}}}` corresponds to some index 
:math:`i \in \{0,\dots,C_{S_{\text{all}}}-1\}` and every state 
:math:`s \in S` to some index :math:`j \in \{0,\dots,N_{S_{\text{all}}}-1\}` and vice versa. 

DTMCs are treated as special MDPs where only a single action exists, which is then enabled in every state, in which case 
:math:`C_{S_{\text{all}}}=N_{S_{\text{all}}}`.

Instantiating DTMCs
===================

DTMCs can be easily instantiated from transition matrices and optional state labelings. Allowed types for the transition matrix include
numpy arrays (or matrices), ordinary 2d-lists and scipy sparse matrices (instances of scipy.sparse.spmatrix).

>>> from switss.model import DTMC
>>> P = [[0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
...      [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
...      [0.0, 0.1, 0.0, 0.7, 0.0, 0.2, 0.0],
...      [0.0, 0.2, 0.0, 0.4, 0.4, 0.0, 0.0],
...      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
...      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
...      [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0]]
>>> labels = { "target" : {3,4,6},
...            "init" :   {0} }
>>> mc = DTMC(P, label_to_states=labels)
>>> mc.digraph().view()
'Digraph.gv.pdf'

Instantiating MDPs
==================

Like DTMCs, MDPs require a transition matrix and optional state labelings. Additional parameters include

* a dictionary (called index_by_state_action) that maps state-action-pairs to their corresponding row-index in the transition matrix,
* and an optional labeling for state-action pairs. 

>>> from switss.model import MDP
>>> index_by_state_action = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (2, 0): 3, (2, 1): 4}
>>> actionlabels = {"A" : { (0,0), (2,0), (1,0) }, "B" : { (2,1), (0,1) } }
>>> P = [[0.3, 0.0, 0.7],
...      [0.0, 1.0, 0.0],
...      [0.5, 0.3, 0.2],
...      [0.8, 0.2, 0.0],
...      [0.0, 0.0, 1.0]]
>>> labels = {  "target": {2},
...             "init"  : {0}}
>>> mdp = MDP(P, index_by_state_action, actionlabels, labels)
>>> mdp.digraph().view()
'Digraph.gv.pdf'

Instantiating from PRISM model files
====================================

SWITSS supports the instantiation of MDPs and DTMCs from .lab and .tra, as well as from PRISM .pm/.nm files:

>>> from switss.model import DTMC
>>> M = DTMC.from_prism_model("datasets/brp.pm",
...                           prism_constants={("N",2),("MAX",1)},
...                           extra_labels={("uncertain","s=5 & srep=2"),("all","true")})
>>> M
DTMC(C=62, N=62, labels={init (1), deadlock (6), uncertain (2), all (62)})

>>> from switss.model import MDP
>>> M = MDP.from_file("datasets/csma-2-2.lab", "datasets/csma-2-2.tra")
>>> M
MDP(C=1054, N=1038, labels={init (1), deadlock (0), all_delivered (3), one_delivered (179), collision_max_backoff (2)})

brp.pm and csma-2-2.lab/.tra can be found in the 
`examples/datasets <https://github.com/simonjantsch/switss/tree/master/examples/datasets>`_ directory.

Rendering of DTMCs and MDPs
===========================

Plotting DTMCs and MDPs is highly customizable. SWITSS implements a `.digraph`-method on DTMCs and MDPs which returns 
`graphviz.Digraph` instances (see `here <https://www.graphviz.org/doc/info/attrs.html>`_). The default behaviour can be changed 
by specifying functions that return graphviz attribute settings for nodes and edges: 

.. code-block::

   from switss.model import DTMC
   M = DTMC.from_prism_model("datasets/crowds.pm",
                          prism_constants={("TotalRuns", 1), ("CrowdSize", 2)},
                          extra_labels={("target","observe0>1")})

   def state_map(stateidx, labels):
      color = "red" if "target" in labels else "blue" if "init" in labels else "green"
      return { "color" : color,
               "label" : "%s [%s]" % (stateidx, ",".join(labels)),
               "style" : "filled" }

   def trans_map(stateidx, targetidx, p):
      return { "color" : "orange", "label" : "{:.2f}%".format(p*100) }

   M.digraph(state_map=state_map, trans_map=trans_map).view()

.. code-block::

   from switss.model import MDP
   M = MDP.from_file("datasets/test.lab", "datasets/test.tra")

   def state_map(stateidx, labels):
      color = "red" if "deadlock" in labels else "blue" if "init" in labels else "green"
      return { "color" : color,
               "label" : "%s [%s]" % (stateidx, ",".join(labels)),
               "style" : "filled" }

   def trans_map(stateidx, action, targetidx, p):
      return { "color" : "orange", "label" : "{:.2f}%".format(p*100) }

   def action_map(sourceidx, action, labels):
      return { "node" : { "label" :  "%s" % action,
                           "color" : "black", 
                           "shape" : "circle" }, 
               "edge" : { "color" : "black",
                           "dir" : "none" } }

   M.digraph(state_map=state_map, trans_map=trans_map, action_map=action_map).view()

crowds.pm as well as test.lab/.tra can be found in the 
`examples/datasets <https://github.com/simonjantsch/switss/tree/master/examples/datasets>`_ directory.
Additional information about graphviz attributes can be found at https://www.graphviz.org/doc/info/attrs.html.

Saving DTMCs and MDPs
=====================

DTMCs and MDPs can be saved as a .tra/.lab files. In order to so, calling `.save(..)` suffices:

>>> from switss.model import DTMC, MDP
>>> d = DTMC([[0,1.],[0,1.]])
>>> d.save("example")
('example.tra', 'example.lab')
>>> m = MDP([[1.0,0.0],[0.5,0.5],[0.0,1.0]], {(0,0) : 0, (0,1) : 1, (1,0) : 2})
>>> m.save("example2")
('example2.tra', 'example2.lab')

Executable examples for the usage of MDPs and DTMCs can be found in `examples/mdp.ipynb <https://github.com/simonjantsch/switss/blob/master/examples/mdp.ipynb>`_,
`examples/dtmc.ipynb <https://github.com/simonjantsch/switss/blob/master/examples/dtmc.ipynb>`_ and 
`examples/custom_graphs.ipynb <https://github.com/simonjantsch/switss/blob/master/examples/custom_graphs.ipynb>`_.

Additional capabilities
=======================

DTMCs and MDPs also support 

* getting labels of states,
* getting states that have some label,
* getting labels of state-actions,
* getting state-actions that have some label,
* computing reachability sets,
* getting actions that are available in some state and
* getting predecessors & successors of states.

Please see the :class:`switss.model.DTMC` and :class:`switss.model.MDP` for more information.

***************************
ReachabilityForm (RF) class
***************************

A RF is a wrapper for DTMCs/MDPs with the following properties:

* exactly one fail, goal and initial state,
* fail and goal have exactly one action, which maps only to themselves,
* the fail state (goal state) has index :math:`N_{S_{\text{all}}}-1` (:math:`N_{S_{\text{all}}}-2`) and the corresponding state-action-pair index :math:`C_{S_{\text{all}}}-1` (:math:`C_{S_{\text{all}}}-2`),
* every state is reachable from the initial state (fail doesn't need to be reachable) and
* every state reaches the goal state (except the fail state).

This kind of DTMC/MDP is one of the core components of SWITSS, since the definitions of Farkas certificates (`FJB19`_, Table 1) can be easily derived thereof.
A further assumption that is needed is that the probability to reach goal or fail is positive under each scheduler.
 
In this context,

* :math:`S = S_{\text{all}} \backslash \{ \text{goal}, \text{fail} \}`,
* :math:`\mathcal{M} = \mathcal{M}_S`,
* :math:`N = N_S` and :math:`C = C_S`,

Furthermore,

* :math:`\textbf{Pr}^{\text{min}}_{s}(\diamond t)` and :math:`\textbf{Pr}^{\text{max}}_{s}(\diamond t)` denote the minimal and maximal probability over all schedulers of eventually reaching state :math:`t` when starting from state :math:`s`. For DTMCs, :math:`\textbf{Pr}^{\text{max}}_{s}(\diamond t) = \textbf{Pr}^{\text{min}}_{s}(\diamond t)`.
* We also define :math:`\textbf{Pr}^{\text{min}}(\diamond t) = (\textbf{Pr}_s^{\text{min}}(\diamond t))_{s \in S}` (respectively for :math:`\text{max}`)  


Reduction of DTMCs/MDPs to ReachabilityForms
============================================

RFs support the reduction of DTMCs/MDPs that do not fulfill the criteria to DTMCs/MDPs that do:

.. code-block::

   from switss.model import DTMC, ReachabilityForm
   M = DTMC.from_file("datasets/crowds-2-3.lab", "datasets/crowds-2-3.tra")
   Mrf, state_map, state_action_map = ReachabilityForm.reduce(M, "init", "target")
   Mrf.system.digraph().view()

In this example, `Mrf.system` is the generated DTMC/MDP in reachability form, `state_map` (`state_action_map`) describes a mapping 
from states (state-action pairs) in `M` to states (state-action pairs) in `Mrf.system`. If a state (state-action pair) does not 
occur in the mapping, it was removed on the way. 

If is also possible to directly instantiate a RF from a DTMC/MDP that already is in reachability form: 

.. code-block::

   from switss.model import DTMC, ReachabilityForm
   M = DTMC.from_file("datasets/crowds-2-3-rf.lab", "datasets/crowds-2-3-rf.tra")
   Mrf = ReachabilityForm(M, "init", "target") 

Reachability probabilities
==========================

RFs implement methods for computing maximal and minimal reachability probabilities :math:`\mathbf{Pr}^{\text{max}}(\diamond goal)` and
:math:`\mathbf{Pr}^{\text{min}}(\diamond goal)`:

.. code-block::

   from switss.model import MDP, ReachabilityForm
   index_by_state_action = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (2, 0): 3, (2, 1): 4, (3,0) : 5}
   actionlabels = {"A" : { (0,0), (2,0), (1,0), (3,0) }, "B" : { (2,1), (0,1) } }

   P = [ [0.3, 0.0, 0.7, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.5, 0.0, 0.0, 0.5],
         [0.5, 0.5, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]

   labels = {  "target": {2},
               "init"  : {0}}

   mdp = MDP(P, index_by_state_action, actionlabels, labels)
   rf,_,_ = ReachabilityForm.reduce(mdp, "init", "target")
   print(rf.pr_max(), rf.pr_min())

************************
Finding small subsystems
************************

In order to search for small subsystems, methods for exact solutions (MILP formulation) and heuristic approaches 
(quotient sum heuristic) are implemented:

.. code-block::

   from switss.model import DTMC, ReachabilityForm
   from switss.problem import MILPExact, QSHeur
   from switss.problem import InverseFrequencyInitializer, InverseReachabilityInitializer

   milpmin = MILPExact(mode="min")
   milpmax = MILPExact(mode="max", solver="gurobi")
   qsheurmin = QSHeur(mode="min", iterations=5)
   qsheurmax = QSHeur(mode="max", solver="cbc")
   qsheurmin_iri = QSHeur(mode="min", solver="gurobi", initializertype=InverseReachabilityInitializer)
   qsheurmax_ifi = QSHeur(mode="max", solver="glpk", initializertype=InverseFrequencyInitializer)

   mc = DTMC.from_file("datasets/crowds-2-3-rf.lab", "datasets/crowds-2-3-rf.tra")
   rf = ReachabilityForm(mc, "init")
   problems = [milpmin, milpmax, qsheurmin, qsheurmax, qsheurmax_ifi, qsheurmin_iri]
   for p in problems:
      result = p.solve(rf, 0.1)
      print(result)
      print("-"*20)

Here, `MILPExact` corresponds to the MILP Formulation and `QSHeur` to the quotient sum heuristic (see :class:`switss.problem.QSHeur`
and :class:`switss.problem.MILPExact` for more information on how to specify additional parameters). 

Results of such minimizations are given as :class:`switss.problem.ProblemResult` instances which contain the objective value of the 
solved MILP/LPs, a :math:`N` or :math:`C` dimensional Farkas certificate (dependent on whether "min" or "max" was choosen) and a 
:class:`switss.problem.Subsystem`-object that contains reachability forms for both super- and subsystem and, additionally, a method
for rendering subsystems with their corresponding certificate values:

.. code-block::
   
   from switss.model import DTMC, ReachabilityForm
   from switss.problem import QSHeur
   mc = DTMC.from_file("datasets/crowds-2-3-rf.lab", "datasets/crowds-2-3-rf.tra")
   rf = ReachabilityForm(mc, "init")
   qs = QSHeur("min")
   result = qs.solve(rf, 0.1)
   result.subsystem.digraph().view()

Label-based minimization
========================

SWITSS also implements label-based minimization, i.e. minimization based on the number of labels in a system. To do so, one can
add lists of labels to `MILPExact` or `QSHeur` instances:

.. code-block::

   from switss.model import DTMC, ReachabilityForm
   from switss.problem import MILPExact

   P = [ [0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.1, 0.0, 0.7, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0],
         [0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0],
         [0.0, 0.2, 0.0, 0.4, 0.2, 0.0, 0.1, 0.1, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.1, 0.2, 0.0],
         [0.0, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0],
         [0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0, 0.1],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0],
         [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

   labels = {  "target" : {8},
               "init" : {0},
               "group1" : {1,3,6},
               "group2" : {7,9,2},
               "group3" : {4,5} }

   mc = DTMC(P, label_to_states=labels)
   rf,_,_ = ReachabilityForm.reduce(mc, "init", "target")
   milp = MILPExact("min")
   result_with_labels = milp.solve(rf, 0.5, labels=["group1","group2","group3"])
   result_no_labels = milp.solve(rf, 0.5)
   print(result_with_labels.subsystem)
   print("---")
   print(result_no_labels.subsystem)

Comparing the results of this particular instance, one will notice that in the second case (`result_no_labels`) the subsystem 
yields a much smaller size than compared with the first instance. In the first case however, label `group2` was completely eliminated.
For exectuable examples, see `examples/groups.ipynb <https://github.com/simonjantsch/switss/blob/master/examples/groups.ipynb>`_.
 
Iterative results
=================

Repeated application of `QSHeur` yields multiple small subsystems along the way. By calling `.solveiter` instead 
of `.solve` on `problem.ProblemFormulation` instances, one can iterate over these solutions. In fact, `.solve` uses `.solveiter` 
itself and returns only the last result:

.. code-block::

   from switss.model import DTMC, ReachabilityForm
   from switss.problem import QSHeur 
   mc = DTMC.from_file("datasets/crowds-2-3-rf.lab", "datasets/crowds-2-3-rf.tra")
   rf = ReachabilityForm(mc)
   qs = QSHeur("min")
   for result in qs.solveiter(rf, 0.01):
      print(result)
      print("-"*10)

************
Certificates
************

SWITSS supports the generation and the checking of Farkas certificates. This can be used, for example, for validating the results 
of solved problem instances: 

>>> from switss.model import DTMC, ReachabilityForm
>>> from switss.certification import generate_farkas_certificate
>>> M = DTMC.from_prism_model("datasets/leader_sync3_2.pm")
>>> Mrf,_,_ = ReachabilityForm.reduce(M,"init","elected")
>>> generate_farkas_certificate(Mrf, "max", ">=", 0.1, solver="cbc")
array([1.3333333 , 0.16666667, 0.16666667, 0.16666667, 0.16666667,
       0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
       0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
       0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
       0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
       1.        ])

>>> from switss.model import DTMC, ReachabilityForm
>>> from switss.problem import MILPExact
>>> M = DTMC.from_prism_model("datasets/leader_sync3_2.pm")
>>> Mrf,_,_ = ReachabilityForm.reduce(M,"init","elected"
>>> from switss.certification import check_farkas_certificate
>>> milp = MILPExact("min")
>>> result = milp.solve(Mrf, 0.1)    
>>> result.farkas_cert
array([0.1, 0. , 0. , 0. , 0. , 0.8, 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0.8, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 1. ])
>>> check_farkas_certificate(Mrf, "min", ">=", 0.1, result.farkas_cert)
True

Supported are checks for all 4 entries of table 1 of [FJB19]_. For more details on both methods, please see `Certification`_. 
For executable examples, see 
`examples/certificates.ipynb <https://github.com/simonjantsch/switss/blob/master/examples/certificates.ipynb>`_.

**********
Benchmarks
**********

In order to make heuristics or exact solutions better comparable, one can run benchmarks on different models, with different 
methods and for varying thresholds. Benchmarks can also be plotted with the help of `matplotlib` for fast visualization:

>>> from switss.problem import QSHeur, MILPExact                                                                                                                                              
>>> from switss.benchmarks import benchmarks as bm                                                                                                                                            
>>> from switss.model import DTMC, ReachabilityForm                                                                                                                                           
>>> import matplotlib.pyplot as plt 
>>> M = DTMC.from_file("datasets/crowds-2-3-rf.lab", "datasets/crowds-2-3-rf.tra")                                                                                                            
>>> Mrf = ReachabilityForm(M,"init")                                                                                                                                                          
>>> qs = QSHeur("min")                                                                                                                                                                        
>>> milp = MILPExact("min")                                                                                                                                                                   
>>> dataqs, datamilp = bm.run(Mrf, [qs,milp], from_thr=0.01, to_thr=0.5, step=0.01)                                                                                                           
>>> fig, ax = plt.subplots(1,1,figsize=(7,6))                                                                                                                                                 
>>> bm.render(dataqs, mode="states-thr", ax=ax, title="QSHeur min vs. MILPExact min")                                                                                                         
<matplotlib.axes._subplots.AxesSubplot object at 0x7fc65d4f3f60>                                                                                                                              
>>> bm.render(datamilp, mode="states-thr", ax=ax)                                                                                                                                             
<matplotlib.axes._subplots.AxesSubplot object at 0x7fc65d4f3f60>
>>> plt.show()

See `examples/benchmarks.ipynb <https://github.com/simonjantsch/switss/blob/master/examples/benchmarks.ipynb>`_ for more executable
examples and `Benchmarking`_ for more details on `run` and `render`.

***************************
Modules & classes reference
***************************

Models
======
.. automodule:: switss.model
   :imported-members:
   :members:
   :undoc-members:

Problem
=======

Note on Initializers and Updaters
---------------------------------

The Initializer and Updater-classes rely on 'groups' of states/state-action pairs for
computing initial/updated objective functions in order unify the concept of label-based and
default minimization. If label-based minimization was chosen, every
group corresponds to some label and thereby to a set of states that have this label.
If label-based minimization was not chosen, every group corresponds to some state or
state-action pair (i.e. every group has only one member).

The objective functions that are returned are given as lists of group-index/group-weight pairings.
For now, define that :math:`V = \{ v_1, \dots, v_m \}` is the set of group indices, i.e. every objective function
assigns a value to all :math:`v \in V`. If a group maps to sets of states (state-action pairs), we 
will write :math:`S_v` (:math:`\mathcal{M}_v`) to indicate this particular set.


.. automodule:: switss.problem
   :imported-members:
   :members:
   :undoc-members:

Certification
=============
.. automodule:: switss.certification
   :imported-members:
   :members:
   :undoc-members:

Benchmarking
============
.. automodule:: switss.benchmarks
   :imported-members:
   :members:
   :undoc-members:

Utils
=====
.. autofunction:: switss.utils.color_from_hash

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

**********
References
**********
.. rubric:: References

.. [FJB19] Funke, F; Jantsch, S; Baier, C: Farkas certificates and minimal witnessing subsystems for probabilistic reachability constraints. (https://link.springer.com/chapter/10.1007/978-3-030-45190-5_18)
