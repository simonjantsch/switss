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
the subsystem is indeed a witness. Work is based on [FJB19]_.


**********************
MDP and DTMC classes
**********************

A MDP is a tuple :math:`\mathcal{M} = (S, \text{Act}, \iota, \textbf{P})`, where 

* :math:`S` denotes the set of states,
* :math:`\text{Act}` denotes the set of actions,
* :math:`\iota` is a probability distribution of initial states on :math:`S`,
* and :math:`\textbf{P}: S \times \text{Act} \times S \rightarrow [0,1]` the transition probability function.
   
Also,

* :math:`\text{Act}(s) \subseteq \text{Act}` denotes the set of actions that can be enabled in state :math:`s` (i.e. :math:`a \in \text{Act}(s) \Leftrightarrow \textbf{P}(s,a,d) > 0` for some state :math:`d`).
* :math:`\mathcal{M} = \{ (s,a) \in S \times \text{Act} \mid a \in \text{Act}(s) \}` denotes the set of state-action-pairs.

For :math:`\textbf{P}` we will use a :math:`C \times N` transition matrix :math:`\text{P}` where :math:`C = | \mathcal{M} |` denotes 
the amount of state-action-pairs and :math:`N = | S |` the amount of states. Furthermore, every state-action pair :math:`(s,a) \in 
\mathcal{M}` corresponds to some index :math:`i = 0,\dots,C-1` and every state :math:`s \in S` to some index :math:`j = 0,\dots,N-1`. 
We define :math:`\text{P}(i,j) := \textbf{P}(s,a,d)` for all indices :math:`i,j` where :math:`i` is the index of :math:`(s,a)` and 
:math:`j` is the index of :math:`d`. 

DTMCs are treated as special MDPs where only a single action exists, which is then enabled in every state, in which case :math:`C=N`.

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
>>> mc = DTMC(P, labels)
>>> mc.digraph().view()
'Digraph.gv.pdf'

Instantiating MDPs
==================

Like DTMCs, MDPs require a transition matrix and optional state labelings. Additional parameters include

* a index_by_state_action dictionary that maps state-action-pairs to their corresponding row-index in the transition matrix,
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

SWITSS supports the instantiation of MDPs and DTMCs from .lab and .tra, as well as from .pm files:

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

In order to make plotting DTMCs and MDPs more customizable, SWITSS implements a `.digraph`-method on DTMCs and MDPs which returns 
`graphviz.Digraph` instances (see `here <https://www.graphviz.org/doc/info/attrs.html>`_). Changing default behaviour can be obtained 
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

Please see the `Models`_ subsection for more information.

***************************
ReachabilityForm (RF) class
***************************

A RF is a special DTMC/MDP with the following properties:

* exactly one fail, goal and initial state :math:`fail,goal,init \in S`,
* fail and goal have only one action :math:`\text{Act}(fail) = \{ a_{fail} \}, \text{Act}(goal) = \{ a_{goal} \}` that maps only to themselves, i.e. :math:`\textbf{P}(fail,a_{fail},fail)=1, \textbf{P}(goal,a_{goal},goal)=1`,
* the fail state (goal state) has index :math:`N-1` (:math:`N-2`) and the corresponding state-action-pair index :math:`C-1` (:math:`C-2`),
* every state is reachable from the initial state (fail doesn't need to be reachable): :math:`\textbf{Pr}_{init}(\diamond s) > 0 \text{ for all } s \in S \backslash \{ fail \}` and
* every state reaches the goal state (except the fail state): :math:`\textbf{Pr}_s(\diamond goal) > 0 \text{ for all } s \in S \backslash \{ fail \}`.


* :math:`\textbf{Pr}^{\text{min}}_{s}(\diamond t)` and :math:`\textbf{Pr}^{\text{max}}_{s}(\diamond t)` denote the minimal and maximal probability over all schedulers of eventually reaching state :math:`t` when starting from state :math:`s`. For DTMCs, :math:`\textbf{Pr}^{\text{max}}_{s}(\diamond t) = \textbf{Pr}^{\text{min}}_{s}(\diamond t)`.
* We also define :math:`\textbf{Pr}^{\text{min}}(\diamond t) = (\textbf{Pr}_s^{\text{min}}(\diamond t))_{s \in S}` (respectively for :math:`\text{max}`)  

This kind of DTMC/MDP is one of the core components of SWITSS. 

********************
Problem Formulations
********************

MILP Formulation
================

Quotient Sum Heuristic
======================

*************
Certification
*************

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

.. [FJB19] Funke, F; Jantsch, S; Baier, C: Farkas certificates and minimal witnessing subsystems for probabilistic reachability constraints. (https://arxiv.org/abs/1910.10636)
