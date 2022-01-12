from switss.model import DTMC, ReachabilityForm
from switss.problem import MILPExact, QSHeur
from switss.certification import generate_farkas_certificate,check_farkas_certificate
import switss.problem.qsheurparams as qsparam
from .example_models import example_dtmcs, toy_dtmc2, toy_dtmc1
import tempfile
import itertools
import warnings

dtmcs = example_dtmcs()
free_lp_solvers = ["cbc","glpk"]
all_lp_solvers = ["cbc","glpk","cplex","gurobi"]
free_milp_solvers = ["cbc"]
all_milp_solvers = ["cbc","gurobi","cplex"]

solvers = free_lp_solvers
milp_solvers = free_milp_solvers

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
    components,proper_mec,mec_count = dtmc.maximal_end_components()
    assert(components[0] == components[1] == components[2])
    assert(len(set([components[0],components[3],components[4],components[5],components[6],components[7]])) == 6)

def test_minimal_witnesses():
    for dtmc in dtmcs:
        reach_form ,_,_ = ReachabilityForm.reduce(dtmc,"init","target")
        instances = [ MILPExact(solver) for solver in milp_solvers ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(dtmc)
            print(threshold)
            results = []
            for mode in ["min", "max"]:
                for instance in instances:
                    results.append(instance.solve(reach_form,threshold,mode,timeout=2))

            # either the status of all results is optimal, or of none of them
            positive_results = [result for result in results if result.status == "optimal"]
            assert len(positive_results) == len(results) or len(positive_results) == 0

            if results[0].status == "optimal":
                # if the result was optimal, the values of all results should be the same
                assert len(set([result.status for result in results])) == 1

def test_label_based_exact_min():
    ex_dtmc = toy_dtmc2()
    reach_form ,_,_ = ReachabilityForm.reduce(ex_dtmc,"init","target")
    instances = [ MILPExact(solver) for  solver in milp_solvers ]
    for threshold in [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
        results = []
        for mode in ["min", "max"]:
            for instance in instances:
                results.append(instance.solve(reach_form,threshold,mode,
                                              labels=["group1","group3"],timeout=20))
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
                reach_form,"min",">=",pr_min_at_init - 1e-8)
            fark_cert_max = generate_farkas_certificate(
                reach_form,"max",">=",pr_max_at_init - 1e-8)

            assert (fark_cert_min is not None) and (fark_cert_max is not None)


def test_heuristics():
    initializers = [qsparam.AllOnesInitializer,
                    qsparam.InverseReachabilityInitializer,
                    qsparam.InverseFrequencyInitializer]

    for dtmc in dtmcs:
        reach_form ,_,_ = ReachabilityForm.reduce(dtmc,"init","target")
        instances = [ QSHeur(iterations=3,initializertype=init,solver=solver)\
                      for (solver,init) in zip(solvers,initializers) ]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.88, 0.9, 0.999, 1,0.9999999999]:
            print(dtmc)
            print(threshold)
            results = []
            for mode in ["min", "max"]:
                for instance in instances:
                    results.append(instance.solve(reach_form,threshold,mode))

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

def test_treelike():
    try:
        from switss.utils.tree_decomp import min_witnesses_from_tree_decomp
        from switss.problem import TreeAlgo
    except:
        warnings.warn(UserWarning("it seems that switss is installed without tree-algo, not running the corresponding tests."))
        return

    for dtmc in [toy_dtmc1(),toy_dtmc2()]:
        rf,_,_ = ReachabilityForm.reduce(dtmc,"init","target")

        for thr in [0.1,0.7,1]:

            instance_tree = TreeAlgo([range(rf.A.shape[0])],solver="gurobi")
            result_tree = instance_tree.solve(rf,thr,"max")

            instance_exact = MILPExact()
            result_exact = instance_exact.solve(rf,thr,"max")
            print(result_tree)
            print(result_exact)

            assert result_tree.value == result_exact.value
