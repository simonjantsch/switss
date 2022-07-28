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
