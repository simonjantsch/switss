from switss.model.mdp import MDP
from switss.model.reachability_form import ReachabilityForm
from switss.problem.qsheur import QSHeur

import stormpy.examples.files
import stormpy

program = stormpy.parse_prism_program(stormpy.examples.files.prism_mdp_maze)

options = stormpy.BuilderOptions()
options.set_build_choice_labels(True)

model = stormpy.build_sparse_model_with_options(program,options)

switss_mdp = MDP.from_stormpy(model)
choice_mdp = MDP.from_stormpy(model,choice_model=True)

N = switss_mdp.N

switss_mdp.add_label(0,"init")
switss_mdp.add_label(N-1,"target")

print(switss_mdp.labels_by_state)

switss_mdp_rf,_,_ = ReachabilityForm.reduce(switss_mdp, "init", "target")

# initialize heuristic to compute a small witness for "max"

qs_heur = QSHeur(solver="cbc",iterations=10)
results = list(qs_heur.solveiter(switss_mdp_rf, 0.00001,"max"))
for r in results:
    print(r.subsystem.subsystem_mask)

