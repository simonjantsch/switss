from switss.model import DTMC, ReachabilityForm
from switss.problem import MILPExact, QSHeur
from switss.certification import generate_farkas_certificate,check_farkas_certificate
import switss.problem.qsheurparams as qsparam
from .example_models import example_dtmcs, toy_dtmc2
import tempfile
import itertools

dtmcs = example_dtmcs()
lp_solvers = ["cbc","gurobi","cplex","glpk"]
solvers = lp_solvers
milp_solvers = ["cbc","gurobi","cplex"]

def test_read_write():
    for dtmc in dtmcs:
        print(dtmc)
        with tempfile.NamedTemporaryFile() as namedtf:
            dtmc.save(namedtf.name)
            read_dtmc = DTMC.from_file(
                namedtf.name + ".lab", namedtf.name + ".tra")

def test_create_reach_form():
    for dtmc in dtmcs:
        print(dtmc)
        reach_form ,_,_ = ReachabilityForm.reduce(dtmc,"init","target")

def test_mecs():
    import numpy as np
    E = [[0,1],[1,2],[2,0],[3,2],[3,1],[3,5],[4,2],[4,6],[5,4],[5,3],[6,4],[7,5],[7,6],[7,7]]
    P = np.zeros(shape=(8,8))
    for u,v in E:
        # initialize with arbitrary probabilities
        ucount = len([w for w,z in E if w == u])
        P[u,v] = 1/ucount
    dtmc = DTMC(P)
    components,mec_count = dtmc.maximal_end_components()
    assert (components == np.array([1., 1., 1., 0., 0., 0., 0., 0.])).all()

def test_minimal_witnesses():
    for dtmc in dtmcs:
        reach_form ,_,_ = ReachabilityForm.reduce(dtmc,"init","target")
        instances = [ MILPExact(mode,solver) for (mode,solver) in zip(["min","max"],milp_solvers) ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(dtmc)
            print(threshold)
            results = []
            for instance in instances:
                results.append(instance.solve(reach_form,threshold,timeout=20))

            # either the status of all results is optimal, or of none of them
            positive_results = [result for result in results if result.status == "optimal"]
            assert len(positive_results) == len(results) or len(positive_results) == 0

            if results[0].status == "optimal":
                # if the result was optimal, tha values of all results should be the same
                assert len(set([result.status for result in results])) == 1

def test_label_based_exact_min():
    ex_dtmc = toy_dtmc2()
    reach_form ,_,_ = ReachabilityForm.reduce(ex_dtmc,"init","target")
    instances = [ MILPExact(mode,solver) for (mode,solver) in zip(["min","max"],milp_solvers) ]
    for threshold in [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
        results = []
        for instance in instances:
            results.append(instance.solve(
                reach_form,threshold,labels=["group1","group3"],timeout=20))
        # either the status of all results is optimal, or of none of them
        positive_results = [result for result in results if result.status == "optimal"]
        assert len(positive_results) == len(results) or len(positive_results) == 0

        if results[0].status == "optimal":
            # if the result was optimal, tha values of all results should be the same
            assert len(set([result.status for result in results])) == 1

def test_certificates():
    for dtmc in dtmcs:
        reach_form ,_,_ = ReachabilityForm.reduce(dtmc,"init","target")
        for sense in ["<","<=",">",">="]:
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
                fark_cert_min = generate_farkas_certificate(
                    reach_form,"min",sense,threshold)
                fark_cert_max = generate_farkas_certificate(
                    reach_form,"max",sense,threshold)
                assert (fark_cert_max is None) == (fark_cert_min is None)
                if fark_cert_max is not None:
                    check_min = check_farkas_certificate(
                        reach_form,"min",sense,threshold,fark_cert_min,tol=1e-5)
                    check_max = check_farkas_certificate(
                        reach_form,"max",sense,threshold,fark_cert_max,tol=1e-5)
                    assert check_min
                    assert check_max


def test_prmin_prmax():
    for dtmc in dtmcs:
        reach_form ,_,_ = ReachabilityForm.reduce(dtmc,"init","target")
        for solver in solvers:
            m_z_st = reach_form.max_z_state(solver=solver)
            m_z_st_act = reach_form.max_z_state_action(solver=solver)
            m_y_st_act = reach_form.max_y_state_action(solver=solver)
            m_y_st = reach_form.max_y_state(solver=solver)

            for vec in [m_z_st,m_z_st_act,m_y_st,m_y_st_act]:
                assert (vec >= -1e-8).all()

            for vec in [m_z_st,m_z_st_act]:
                assert (vec <= 1+1e-8).all()

            pr_min = reach_form.pr_min()
            pr_max = reach_form.pr_max()

            for vec in [pr_min,pr_max]:
                assert (vec <= 1).all() and (vec >= 0).all()

            assert (pr_min == pr_max).all()

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

    for dtmc in dtmcs:
        reach_form ,_,_ = ReachabilityForm.reduce(dtmc,"init","target")
        instances = [ QSHeur(mode,iterations=3,initializertype=init,solver=solver)\
                      for (mode,solver,init) in zip(["min","max"],solvers,initializers) ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(dtmc)
            print(threshold)
            results = []
            for instance in instances:
                results.append(instance.solve(reach_form,threshold))

            # either the status of all results is optimal, or of none of them
            positive_results = [result for result in results if result.status == "optimal"]
            assert len(positive_results) == len(results) or len(positive_results) == 0

            # test the construction of the resulting subsystems
            for r in results:
                if r.status == "optimal":
                    assert r.value >= 0
                    ss_mask = r.subsystem.subsystem_mask
                    ss_reach_form = r.subsystem.reachability_form
                    super_reach_form = r.subsystem.supersys_reachability_form
                    ss_model = r.subsystem.model
