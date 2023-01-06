from switss.model import MDP, ReachabilityForm, RewardReachabilityForm
from switss.problem import MILPExact, QSHeur
from switss.certification import generate_farkas_certificate,check_farkas_certificate
import switss.problem.qsheurparams as qsparam
import tempfile
from switss.model import MDP,DTMC,ReachabilityForm

def example_dtmcs():
    dtmcs = []
    dtmcs.append(
        DTMC.from_prism_model("./examples/datasets/crowds.pm",
                              prism_constants={("CrowdSize",2),("TotalRuns",8)},
                              extra_labels={("target","observe0>1")}))
    dtmcs.append(toy_dtmc1())
    dtmcs.append(toy_dtmc2())
    dtmcs.append(DTMC.from_prism_model(
        "./examples/datasets/leader_sync3_2.pm",
        extra_labels={("target","s1=3 & s2=3 & s3=3")}))
    dtmcs.append(
        DTMC.from_prism_model("./examples/datasets/brp.pm",
                              prism_constants={("N",2),("MAX",1)},
                              extra_labels={("target","s=5 & srep=2"),
                                            ("all","true")}))

    return dtmcs

def example_mdps():
    mdps = []
    mdps.append(toy_mdp1())
    mdps.append(toy_mdp2())
    mdps.append(MDP.from_prism_model("./examples/datasets/csma2_2.nm",
                                     extra_labels={("target","s1=4&s2=4")}
                                     ))
    mdps.append(
        MDP.from_prism_model("./examples/datasets/coin2.nm",
                             prism_constants = {("K",2)},
                             extra_labels={("target","pc1=3 & pc2=3")}
                             ))
    #only this last mdp has proper end components
    mdps.append(toy_mdp3())
    return mdps

def toy_mdp1():
    index_by_state_action = {
        (0, 0): 0, (0, 1): 1, (1, 0): 2, (2, 0): 3, (2, 1): 4}
    actionlabels = {
        "A" : { (0,0), (2,0), (1,0) }, "B" : { (2,1), (0,1) } }

    P = [[0.3, 0.0, 0.7],
         [0.0, 1.0, 0.0],
         [0.5, 0.3, 0.2],
         [0.8, 0.2, 0.0],
         [0.0, 0.0, 1.0]]

    labels = {  "target": {2}, "rewtarget" : {2}, "init"  : {0}}

    rewards = [0,0,0,0,0]

    return(MDP(P, index_by_state_action, actionlabels, labels, reward_vector= rewards))

def toy_mdp2():

    index_by_state_action = {(0,0) : 0, (0,1) : 1, (1,0) : 2, (2,0) : 3, (3,0) : 4, (4,0) : 5, (5,0) : 6, (6,0) : 7, (7,0) : 8}

    P = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, 0.2],
         [0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.3],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

    label_to_states = {    "target" : {6},
                           "fail" : {7},
                           "rewtarget" : {6,7},
                           "init" : {0},
                           "blue" : {1,2,3},
                           "lightblue"  : {4},
                           "brown" : {5}
                      }

    rewards = [7,0.22,1.6,0,0,1,23,11,9]

    return MDP(P,index_by_state_action,dict([]),label_to_states,reward_vector=rewards)

#with a proper end component
def toy_mdp3():
    index_by_state_action = {
        (0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3, (2, 0): 4}
    actionlabels = {
        "A" : { (0,0), (2,0), (1,0) }, "B" : { (1,1), (0,1) } }

    P = [[0.3, 0.0, 0.7],
         [0.0, 1.0, 0.0],
         [0.5, 0.3, 0.2],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]

    labels = {  "target": {2}, "rewtarget" : {2}, "init"  : {0}}

    rewards = [2,1,0.6,0,0]

    return(MDP(P, index_by_state_action, actionlabels, labels, reward_vector=rewards))

def toy_mdp4():
    index_by_state_action = {
        (0, 0): 0, (0, 1): 1, (1, 0): 2, (2, 0): 3, (3, 0): 4, (3,1) :5, (4,0) : 6, (5,0) : 7}
    actionlabels = {
        "A" : { (0,0), (2,0), (1,0) }, "B" : { (1,1), (0,1) } }

    P = [[0.0, 0.0, 0.75, 0.25, 0.00, 0.0],
         [0.0, 1.0, 0.00, 0.00, 0.00, 0.0],
         [0.5, 0.0, 0.50, 0.00, 0.00, 0.0],
         [0.0, 0.0, 0.25, 0.50, 0.25, 0.0],
         [0.0, 1.0, 0.00, 0.00, 0.00, 0.0],
         [0.0, 0.0, 0.00, 0.00, 0.40, 0.6],
         [0.0, 0.0, 0.00, 0.00, 1.00, 0.0],
         [0.0, 0.0, 0.00, 0.00, 0.00, 1.0]]

    labels = {  "target": {4}, "rewtarget" : {4,5}, "init"  : {0}}

    rewards = [0,0,0,0,0,0,0,0]

    return(MDP(P, index_by_state_action, actionlabels, labels, reward_vector=rewards))

def toy_dtmc1():
    P = [[0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
         [0.0, 0.1, 0.0, 0.7, 0.0, 0.2, 0.0],
         [0.0, 0.2, 0.0, 0.4, 0.4, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0]]

    labels = { "target" : {3,4,6}, "init" : {0} }

    return(DTMC(P, label_to_states=labels))

def toy_dtmc2():
    P = [[0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.1, 0.0, 0.7, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0],
         [0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0],
         [0.0, 0.2, 0.0, 0.4, 0.2, 0.0, 0.1, 0.1, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.1, 0.2, 0.0],
         [0.0, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0],
         [0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0, 0.1],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0],
         [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    labels = {"target" : {8},
              "init" : {0},
              "group1" : {1,3,6},
              "group2" : {7,9,2},
              "group3" : {4,5} }

    return(DTMC(P, label_to_states=labels))

def toy_dtmc3():
    P = [[0.1,0.9,0],
         [0,0.2,0.8],
         [0,1,0]]

    return(DTMC(P))


mdps = example_mdps()

free_lp_solvers = ["cbc","glpk"]
free_milp_solvers = ["cbc"]
all_lp_solvers = ["cbc","gurobi","cplex","glpk"]
all_milp_solvers = ["cbc","gurobi","cplex"]

lp_solvers = free_lp_solvers
milp_solvers = free_milp_solvers

def test_minimal_witnesses():
    # only test the first 3 examples, as the others are too large
    i = 0
    for mdp in [toy_mdp1(), toy_mdp2(), toy_mdp4()]:
        print("creating new reach form")
        reach_form ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        instances = [ MILPExact(solver) for solver in milp_solvers ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.56, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(mdp)
            print(threshold)
            #print(mdp.digraph())
            max_results, min_results = [], []
            for instance in instances:
                min_results.append(instance.solve(reach_form,threshold, "min"))
                max_results.append(instance.solve(reach_form,threshold, "max"))

            # either the status of all results is optimal, or of none of them
            # and this should hold for min and max
            min_positive_results = [result for result in min_results if result.status == "optimal"]
            assert len(min_positive_results) == len(min_results) or len(min_positive_results) == 0

            max_positive_results = [result for result in max_results if result.status == "optimal"]
            assert len(max_positive_results) == len(max_results) or len(max_positive_results) == 0

            if min_results[0].status == "optimal":
                assert len(set([result.status for result in min_results])) == 1
                assert max_results[0].status == "optimal"
                assert len(set([result.status for result in max_results])) == 1
                assert max_results[0].value <= min_results[0].value
            elif max_results[0].status == "optimal":
                assert max_results[0].status == "optimal"
                assert len(set([result.status for result in max_results])) == 1

            if i == 1:
                if threshold == 0.1:
                    assert max_results[0].value == 3
                    assert min_results[0].value == 5
                elif threshold == 0.3:
                    assert max_results[0].value == 4
                    assert min_results[0].value == 7
                elif threshold == 0.56:
                    assert max_results[0].value == 5
        i += 1

test_minimal_witnesses()
