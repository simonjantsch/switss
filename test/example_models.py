from farkas.model import MDP,DTMC,ReachabilityForm

def example_dtmcs():
    dtmcs = []
    dtmcs.append(
        DTMC.from_prism_model("./examples/datasets/crowds.pm",
                              prism_constants={("CrowdSize",2),("TotalRuns",3)},
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
    mdps.append(MDP.from_file("./examples/datasets/csma-2-2.lab",
                              "./examples/datasets/csma-2-2.tra"))
    mdps.append(
        MDP.from_prism_model("./examples/datasets/coin2.nm",
                             extra_labels={("target","pc1=3 & pc2=3")}))
    mdps.append(toy_mdp1())
    return mdps

def toy_mdp1():
    index_by_state_action = {
        (0, 0): 0, (0, 1): 1, (1, 0): 2, (2, 0): 3, (2, 1): 4}
    actionlabels = {
        "A" : { (0,0), (2,0), (1,0) }, "B" : { (2,1), (0,1) } }

    P = [[0.3, 0.0, 0.7],
         [0.0, 1.0, 0.0],
         [0.5, 0.5, 0.0],
         [0.8, 0.2, 0.0],
         [0.0, 0.0, 1.0]]

    labels = {  "target": {2}, "init"  : {0}}

    return(MDP(P, index_by_state_action, actionlabels, labels))



def toy_dtmc1():
    P = [[0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
         [0.0, 0.1, 0.0, 0.7, 0.0, 0.2, 0.0],
         [0.0, 0.2, 0.0, 0.4, 0.4, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0]]

    labels = { "target" : {3,4,6}, "init" : {0} }

    return(DTMC(P, labels))

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

    return(DTMC(P, labels))

