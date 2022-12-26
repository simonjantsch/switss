from switss.model import MDP, ReachabilityForm, RewardReachabilityForm
from switss.problem import MILPExact, QSHeur
from switss.certification import generate_farkas_certificate,check_farkas_certificate
import switss.problem.qsheurparams as qsparam
from .example_models import example_mdps, toy_mdp2, toy_mdp1, toy_mdp3, toy_mdp4
import tempfile

mdps = example_mdps()

free_lp_solvers = ["cbc","glpk"]
free_milp_solvers = ["cbc"]
all_lp_solvers = ["cbc","gurobi","cplex","glpk"]
all_milp_solvers = ["cbc","gurobi","cplex"]

lp_solvers = free_lp_solvers
milp_solvers = free_milp_solvers

def test_read_write():
    for mdp in mdps:
        print(mdp)
        with tempfile.NamedTemporaryFile() as namedtf:
            mdp.save(namedtf.name)
            read_mdp = MDP.from_file(
                namedtf.name + ".lab", namedtf.name + ".tra")

def test_create_reach_form():
    for mdp in mdps:
        print(mdp)
        reach_form ,_,_ = ReachabilityForm.reduce(mdp,"init","target")

def test_mecs():
    import numpy as np
    SAPpairs = [(0,0,1,1),(1,0,2,1),(2,0,0,1),(3,0,2,1),(4,0,2,0.5),(3,1,5,1),(5,1,3,1),(5,0,4,1),(4,0,6,0.5),(6,0,4,1),(7,1,6,0.5),(7,1,5,0.5),(7,0,7,1)]
    index_by_state_action = {(0,0):0,(1,0):1,(2,0):2,(3,0):3,(4,0):4,(3,1):5,(5,1):6,(5,0):7,(6,0):8,(7,1):9,(7,0):10}
    P = np.zeros(shape=(11,8))
    for s,a,d,p in SAPpairs:
        P[index_by_state_action[(s,a)],d] = p
    mdp = MDP(P,index_by_state_action)
    components,_,mec_count = mdp.maximal_end_components()
    assert (components[0] == components[1] == components[2])
    assert (components[3] == components[5])
    assert(len(set(components)) == 5)

def test_proper_mecs():
    rf,_,_ = ReachabilityForm.reduce(toy_mdp3(),"init","target")
    assert (not rf.in_proper_ec(0) and rf.in_proper_ec(1) and not rf.in_proper_ec(2))

def test_mec_free():
    for mdp in mdps[:-1]:
        rf ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        rf._check_mec_freeness()

def test_minimal_witnesses():
    # only test the first 3 examples, as the others are too large
    i = 0
    for mdp in [toy_mdp1(),toy_mdp2(),toy_mdp4()]: #toy_mdp3()
        reach_form ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        instances = [ MILPExact(solver) for solver in milp_solvers ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.56, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(mdp)
            print(threshold)
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


def test_minimal_proper_ecs():
    reach_form ,_,_ = ReachabilityForm.reduce(toy_mdp3(),"init","target")
    instance = MILPExact("cbc")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
        min_results = []
        min_results.append(instance.solve(reach_form,threshold, "min"))

def test_label_based_exact_min():
    ex_mdp = toy_mdp2()
    reach_form ,_,_ = ReachabilityForm.reduce(ex_mdp,"init","target")
    instances = [ MILPExact(solver) for solver in milp_solvers ]
    for threshold in [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
        min_results, max_results = [],[]
        for instance in instances:
            min_results.append(instance.solve(reach_form,threshold,"min",labels=["blue"]))
            max_results.append(instance.solve(reach_form,threshold,"max",labels=["blue"]))
        # either the status of all results is optimal, or of none of them
        min_positive_results = [result for result in min_results if result.status == "optimal"]
        assert len(min_positive_results) == len(min_results) or len(min_positive_results) == 0
        max_positive_results = [result for result in max_results if result.status == "optimal"]
        assert len(max_positive_results) == len(max_results) or len(max_positive_results) == 0

        if min_results[0].status == "optimal":
            # if the result was optimal, tha values of all results should be the same
            assert len(set([result.status for result in min_results])) == 1
            assert max_results[0].status == "optimal"
            assert max_results[0].value <= min_results[0].value

        elif max_results[0].status == "optimal":
            assert len(set([result.status for result in max_results])) == 1

def test_certificates():
    for mdp in mdps:
        reach_form ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        for sense in ["<","<=",">",">="]:
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1]:
                print(mdp)
                print(threshold)
                fark_cert_min = generate_farkas_certificate(
                    reach_form,"min",sense,threshold)
                fark_cert_max = generate_farkas_certificate(
                    reach_form,"max",sense,threshold)
                # if the sense is >= or > and there is a certificate for pr_min, then there is one for pr_max
                if sense in [">",">="]:
                    assert (fark_cert_min is None) or (fark_cert_max is not None)
                # and dually:
                if sense in ["<","<="]:
                    assert (fark_cert_max is None) or (fark_cert_min is not None)
                if sense in [">",">="] and fark_cert_min is not None:
                    check_min = check_farkas_certificate(
                        reach_form,"min",sense,threshold,fark_cert_min,tol=1e-5)
                    check_max = check_farkas_certificate(
                        reach_form,"max",sense,threshold,fark_cert_max,tol=1e-5)
                    assert check_min
                    assert check_max
                if sense in ["<","<="] and fark_cert_max is not None:
                    check_min = check_farkas_certificate(
                        reach_form,"min",sense,threshold,fark_cert_min,tol=1e-5)
                    check_max = check_farkas_certificate(
                        reach_form,"max",sense,threshold,fark_cert_max,tol=1e-5)
                    assert check_min
                    assert check_max

def test_prmin_prmax():
    for mdp in mdps:
        reach_form ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        for solver in lp_solvers:
            m_z_st = reach_form.max_z_state(solver=solver)
            m_z_st_act = reach_form.max_z_state_action(solver=solver)

            if reach_form.is_ec_free:
                m_y_st_act = reach_form.max_y_state_action(solver=solver)
                m_y_st = reach_form.max_y_state(solver=solver)
                assert (m_y_st_act >= -1e-8).all()
                assert (m_y_st >= -1e-8).all()

            for vec in [m_z_st,m_z_st_act]:
                assert (vec >= -1e-8).all()

            for vec in [m_z_st,m_z_st_act]:
                assert (vec <= 1+1e-8).all()

            pr_min = reach_form.pr_min()
            pr_max = reach_form.pr_max()

            for vec in [pr_min,pr_max]:
                assert (vec <= 1).all()
                if reach_form.is_ec_free:
                    assert (vec > 0).all()

            assert (pr_min <= pr_max).all()

            pr_min_at_init = pr_min[reach_form.initial]
            pr_max_at_init = pr_max[reach_form.initial]

            # we find farkas certificates for pr_min and pr_max
            fark_cert_min = generate_farkas_certificate(
                reach_form,"min",">=",pr_min_at_init)
            fark_cert_max = generate_farkas_certificate(
                reach_form,"max",">=",pr_max_at_init)

            assert (fark_cert_min is not None) and (fark_cert_max is not None)

def test_heuristics():
    initializers = [qsparam.AllOnesInitializer,
                    qsparam.InverseReachabilityInitializer,
                    qsparam.InverseFrequencyInitializer]

    for mdp in mdps:
        reach_form ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        instances = [ QSHeur(iterations=3,initializertype=init,solver=solver) \
                      for (solver,init) in zip(lp_solvers,initializers) ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(mdp)
            print(threshold)
            max_results, min_results = [], []
            for instance in instances:
                min_results.append(instance.solve(reach_form,threshold,"min"))
                max_results.append(instance.solve(reach_form,threshold,"max"))
            # either the status of all results is optimal, or of none of them
            min_positive_results = [result for result in min_results if result.status == "optimal"]
            assert len(min_positive_results) == len(min_results) or len(min_positive_results) == 0
            max_positive_results = [result for result in max_results if result.status == "optimal"]
            assert len(max_positive_results) == len(max_results) or len(max_positive_results) == 0

            # test the construction of the resulting subsystems
            for r in min_results + max_results:
                if r.status == "optimal":
                    assert r.value >= 0
                    ss_mask = r.subsystem.subsystem_mask
                    ss_reach_form = r.subsystem.reachability_form
                    super_reach_form = r.subsystem.supersys_reachability_form
                    ss_model = r.subsystem.model

def test_rewards_heur():
    initializers = [qsparam.AllOnesInitializer,
                    qsparam.InverseReachabilityInitializer,
                    qsparam.InverseFrequencyInitializer]

    for mdp in mdps:
        if mdp.reward_vector is None:
            continue

        reach_form ,_,_ = ReachabilityForm.reduce(mdp,"init","rewtarget")

        if not reach_form.is_ec_free:
            continue

        reward_reach_form,_,_ = RewardReachabilityForm.reduce(mdp,"init","rewtarget")

        instances = [ QSHeur(iterations=3,initializertype=init,solver=solver) \
                      for (solver,init) in zip(lp_solvers,initializers) ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(mdp)
            print(threshold)
            max_results, min_results = [], []
            for instance in instances:
                min_results.append(instance.solve(reward_reach_form,threshold,"min"))
                max_results.append(instance.solve(reward_reach_form,threshold,"max"))
            # either the status of all results is optimal, or of none of them
            min_positive_results = [result for result in min_results if result.status == "optimal"]
            assert len(min_positive_results) == len(min_results) or len(min_positive_results) == 0
            max_positive_results = [result for result in max_results if result.status == "optimal"]
            assert len(max_positive_results) == len(max_results) or len(max_positive_results) == 0

            # test the construction of the resulting subsystems
            for r in min_results + max_results:
                if r.status == "optimal":
                    assert r.value >= 0
                    ss_mask = r.subsystem.subsystem_mask
                    ss_reward_reach_form = r.subsystem.reachability_form
                    super_reward_reach_form = r.subsystem.supersys_reachability_form
                    ss_model = r.subsystem.model

def test_rewards_exact():
    for mdp in mdps[:1]:
        reach_form ,_,_ = RewardReachabilityForm.reduce(mdp,"init","rewtarget")
        instances = [ MILPExact(solver) for solver in milp_solvers ]
        for threshold in [0.1,2,5,10]:
            print(mdp)
            print(threshold)
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

