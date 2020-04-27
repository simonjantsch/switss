# TODO
# function for running benchmarks
# display results
# save results as .csv 
# Features: timing

# display modes:
#   #states vs. threshold
#   time vs. threshold
#   milpexact vs qsheur (?)
#   groups ???

from ..model import ReachabilityForm
from ..problem import ProblemFormulation

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

def run(reachability_form, method, from_thr=1e-3, to_thr=1, step=1e-3, debug=False):
    assert isinstance(reachability_form, ReachabilityForm)
    assert isinstance(method, ProblemFormulation)
    thresholds = np.arange(from_thr, to_thr+step, step)
    data = { "method" : method.details , "run" : [] }
    for thr in thresholds:
        starttime = timer()
        els = []
        for result in method.solveiter(reachability_form, thr):
            time = timer() - starttime
            statecount = np.sum(result.subsystem.subsystem_mask)
            el = {  "threshold" : thr,
                    "statecount" : statecount,
                    "time" : time }
            if debug:
                print(",".join(["%s=%s" % it for it in el.items()]))
            els.append(el)
            starttime = timer()
        data["run"].append(els)
    return data

def render(run, mode="states-thr", ax=None):
    assert mode in ["time-thr", "states-thr"]
    if ax is None:
        ax = plt.subplot()
    
    resultcount = len(run["run"][0])
    if mode == "states-thr":
        plt.ylabel("states")
        for idx in range(resultcount):
            thr = [el[idx]["threshold"] for el in run["run"]]
            label = r"%s" % run["method"]["type"] if resultcount == 1 else r"$%s_{%s}$" % (run["method"]["type"], idx+1)
            # label=",".join(["\n%s=%s" % it for it in run["method"].items()])
            sta = [el[idx]["statecount"] for el in run["run"]]
            ax.plot(thr, sta, linestyle="dashed", marker="x", label=label)
    elif mode == "time-thr":
        plt.ylabel("time")
        tim = np.array([el[0]["time"] for el in run["run"]])
        thr = [el[0]["threshold"] for el in run["run"]]
        label = r"%s" % run["method"]["type"]
        for idx in range(1,resultcount):
            tim = np.vstack((tim,[el[idx]["time"] for el in run["run"]]))
        tim = np.sum(tim,axis=0)
        ax.plot(thr, tim, linestyle="dashed", marker="x",  label=label)
    plt.xlabel(r"threshold $\lambda$")
    plt.title(run["method"]["type"])
    plt.grid()
    plt.legend()
    return ax