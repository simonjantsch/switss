from switss.model import MDP, ReachabilityForm
from switss.problem import MILPExact, QSHeur
from switss.certification import generate_farkas_certificate,check_farkas_certificate
import switss.problem.qsheurparams as qsparam
from .example_models import example_mdps, toy_mdp2
import tempfile

mdps = example_mdps()
lp_solvers = ["cbc","gurobi","cplex","glpk"]
milp_solvers = ["cbc","gurobi","cplex"]

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

def test_mec_free():
    for mdp in mdps:
        rf ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        rf._check_mec_freeness()

        # rf ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        # new_label_to_states = rf.system.states_by_label
        # for st in rf.system.labels_by_state.keys():
        #     if st in new_label_to_states["fail"]:
        #         new_label_to_states.add("target",st)
        # target_or_fail_mdp = MDP(
        #     rf.system.P,rf.system.index_by_state_action,{},new_label_to_states)
        # target_or_fail_rf ,_,_ = ReachabilityForm.reduce(target_or_fail_mdp,"init","target")
        # assert (target_or_fail_rf.pr_min() == 1).all()

def test_minimal_witnesses():
    # only test the first 2 examples, as the others are too large
    for mdp in mdps[:1]:
        reach_form ,_,_ = ReachabilityForm.reduce(mdp,"init","target")
        min_instances = [ MILPExact("min",solver) for solver in milp_solvers ]
        max_instances = [ MILPExact("max",solver) for solver in milp_solvers ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(mdp)
            print(threshold)
            min_results = []
            for instance in min_instances:
                min_results.append(instance.solve(reach_form,threshold))

            max_results = []
            for instance in max_instances:
                max_results.append(instance.solve(reach_form,threshold))

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
                assert len(set([result.status for result in maxresults])) == 1

def test_label_based_exact_min():
    ex_mdp = toy_mdp2()
    reach_form ,_,_ = ReachabilityForm.reduce(ex_mdp,"init","target")
    min_instances = [ MILPExact("min",solver) for solver in milp_solvers ]
    max_instances = [ MILPExact("max",solver) for solver in milp_solvers ]
    for threshold in [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
        min_results, max_results = [],[]
        for instance in min_instances:
            min_results.append(instance.solve(
                reach_form,threshold,labels=["blue"]))
        for instance in max_instances:
            max_results.append(instance.solve(
                reach_form,threshold,labels=["blue"]))
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
            m_y_st_act = reach_form.max_y_state_action(solver=solver)
            m_y_st = reach_form.max_y_state(solver=solver)

            for vec in [m_z_st,m_z_st_act,m_y_st,m_y_st_act]:
                assert (vec >= 0).all

            for vec in [m_z_st,m_z_st_act]:
                assert (vec <= 1).all

            pr_min = reach_form.pr_min()
            pr_max = reach_form.pr_max()

            for vec in [pr_min,pr_max]:
                assert (vec <= 1).all and (vec > 0).all

            assert (pr_min <= pr_max).all

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
        min_instances = [ QSHeur("min",iterations=3,initializertype=init,solver=solver)\
                          for (solver,init) in zip(lp_solvers,initializers) ]
        max_instances = [ QSHeur("max",iterations=3,initializertype=init,solver=solver)\
                          for (solver,init) in zip(lp_solvers,initializers) ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(mdp)
            print(threshold)
            max_results, min_results = [], []
            for instance in min_instances:
                min_results.append(instance.solve(reach_form,threshold))
            for instance in max_instances:
                max_results.append(instance.solve(reach_form,threshold))
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
