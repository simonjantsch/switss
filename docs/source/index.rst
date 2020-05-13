.. switss documentation master file, created by
   sphinx-quickstart on Mon Mar 16 15:04:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SWITSS
========

A tool for the computation of Small WITnessing SubSystems in Markov Decision Processes (MDPs) and 
Discrete Time Markov Chains (DTMCs). SWITSS implements exact and heuristic methods for computing small 
witnessing subsystems by reducing the problem to (mixed integer) linear programming. Returned subsystems 
can automatically be rendered graphically and are accompanied with a certificate which proves that 
the subsystem is indeed a witness. Work is based on [FJB19]_.

We will use the following notation:

* a MDP is a tuple :math:`\mathcal{M} = (S, \text{Act}, \iota, \textbf{P})`, where 
   * :math:`S` denotes the set of states,
   * :math:`\text{Act}` denotes the set of actions,
   * :math:`\iota` is a probability distribution of initial states on :math:`S`,
   * and :math:`\textbf{P}: S \times \text{Act} \times S \rightarrow [0,1]` the transition probability function.

For :math:`\textbf{P}` we will use a :math:`C \times N` transition matrix where :math:`C` denotes the amount of state-action-pairs 
and :math:`N` the amount of states. Let :math:`(i,j) \in \{0,\dots,C-1\} \times \{0,\dots,N-1\}` where :math:`i` corresponds to some 
state-action-pair :math:`(s,a) \in S \times \text{Act}` and :math:`j` to some state :math:`d \in S`. :math:`\textbf{P}(i,j)` then 
is the probability of transitioning to state :math:`d` when action :math:`a` is taken in state :math:`s`.

DTMCs are treated as special MDPs where only a single action exists, which is then enabled in every state, and therefore :math:`C=N`.

Also,

* :math:`\text{Act}(s) \subseteq \text{Act}` denotes the set of actions that can be enabled in state :math:`s`.
* :math:`\textbf{Pr}^{\text{min}}_{s}(\diamond t)` and :math:`\textbf{Pr}^{\text{max}}_{s}(\diamond t)` denote the minimal and maximal probability over all schedulers of eventually reaching state :math:`t` when starting from state :math:`s`. For DTMCs, :math:`\textbf{Pr}^{\text{max}}_{s}(\diamond t) = \textbf{Pr}^{\text{min}}_{s}(\diamond t)`.
* :math:`\textbf{Pr}^{\text{min}}(\diamond t) = (\textbf{Pr}_s^{\text{min}}(\diamond t))_{s \in S}` (respectively for :math:`\text{max}`)  

And finally,

* :math:`\mathcal{P}` denotes the power set.

Modules
=======

Models
------
.. automodule:: switss.model
   :imported-members:
   :members:
   :undoc-members:

Prism
------
.. automodule:: switss.prism
   :imported-members:
   :members:
   :undoc-members:

Utils
------
.. automodule:: switss.utils
   :imported-members:
   :members:
   :undoc-members:

Solver
-------
.. automodule:: switss.solver
   :imported-members:
   :members:
   :undoc-members:

Problem
-------
.. automodule:: switss.problem
   :imported-members:
   :members:
   :undoc-members:

Certification
-------------
.. automodule:: switss.certification
   :imported-members:
   :members:
   :undoc-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==================
.. rubric:: References

.. [FJB19] Funke, F; Jantsch, S; Baier, C: Farkas certificates and minimal witnessing subsystems for probabilistic reachability constraints. (https://arxiv.org/abs/1910.10636)
