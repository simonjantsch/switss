from farkas.model import DTMC, ReachabilityForm
from farkas.problem.milpexact import MILPExact
from farkas.certification import generate_farkas_certificate,check_farkas_certificate
from .example_models import example_dtmcs
import tempfile

dtmcs = example_dtmcs()

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
        reach_form = ReachabilityForm(dtmc,"init","target")

def test_minimal_witnesses():
    for dtmc in dtmcs:
        reach_form = ReachabilityForm(dtmc,"init","target")
        threshold = 0
        for i in range(11):
            print(dtmc)
            print(threshold)
            exact_min = MILPExact(threshold,"min")
            exact_max = MILPExact(threshold,"max")
            result_min = exact_min.solve(reach_form)
            result_max = exact_max.solve(reach_form)
            threshold = threshold + 0.1

            assert result_min.status == result_max.status
            if result_min.status == "optimal":
                assert result_min.value == result_max.value

def test_certificates():
    for dtmc in dtmcs:
        reach_form = ReachabilityForm(dtmc,"init","target")
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
