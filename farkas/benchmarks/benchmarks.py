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
from collections.abc import Iterable

def run(reachability_form, method, from_thr=1e-3, to_thr=1, step=1e-3, debug=False):
    """Runs a benchmark test of a method on a given reachability form. The benchmark consists
    of running the method on the reachability form for varying thresholds. Returns a dictionary which contains
    result of the specified test. `from_thr` and `to_thr` specify the smallest and greatest 
    threshold respectively. `step` specifies the resolution (i.e. distance between neighbouring thresholds).

    It is also possible to define multiple methods by giving an iterable (like a list). If 
    that is the case, a list of the generated dataset for each method is returned.

    If method is not an iterable but an instance of a ProblemFormulation, the output is given as a dictionary
    of the form

    .. code-block::

        { "method" : { "type" : method_type, ... },
          "run" : [ { "threshold" : threshold,
                      "statecounts" : [statecount1,statecount2,...]
                      "times" : [time1,time2,...] },
                     ...] }
    
    where "method" contains information about the used method (see problem.ProblemUtils.details) and
    "run" contains a list of 

    :param reachability_form: The given reachability form.
    :type reachability_form: model.ReachabilityForm
    :param method: A problem formulation that is evalutated in this benchmark.
    :type method: problem.ProblemFormulation
    :param from_thr: Smallest threshold, defaults to 1e-3
    :type from_thr: float, optional
    :param to_thr: Greatest threshold, defaults to 1
    :type to_thr: float, optional
    :param step: Distance between neighbouring thresholds, defaults to 1e-3
    :type step: float, optional
    :param debug: If True, additional output is printed to the console, defaults to False
    :type debug: bool, optional
    :return: The generated data.
    :rtype: Dict or List
    """    
    assert isinstance(reachability_form, ReachabilityForm)
    
    if isinstance(method, Iterable):
        ret = []
        for idx, m in enumerate(method):
            if debug:
                print("="*50)
                print("running benchmark %s/%s" % (idx+1, len(method)))
            data = run(reachability_form, m, from_thr, to_thr, step, debug)
            ret.append(data)
        return ret

    assert isinstance(method, ProblemFormulation)

    thresholds = np.arange(from_thr, min(1,to_thr+step), step)
    data = { "method" : method.details , "run" : [] }
    if debug:
        print("-"*50)
        print("%s" % "\n".join(["%s=%s" % it for it in method.details.items()]))
        print("-"*50)
    for idx,thr in enumerate(thresholds):
        p = (idx+1)/len(thresholds)
        starttime = timer()
        times, statecounts = [], []
        for result in method.solveiter(reachability_form, thr):
            time = timer() - starttime
            statecount = np.sum(result.subsystem.subsystem_mask)
            statecounts.append(statecount)
            times.append(time)
            starttime = timer()
        if debug:
            print("\tp={:.3f} threshold={:.3f} statecount={} time={:.3f}".format(p,thr,statecounts[-1], sum(times)) )
        els = { "threshold" : thr, "statecounts" : statecounts, "times" : times }
        data["run"].append(els)
    return data

def render(run, mode="laststates-thr", ax=None, title=None):
    """Renders a benchmark run via matplotlib. `mode` specifies the type of the
    resulting plot, i.e. statecount vs. threshold ('states-thr', plots all intermediate results), only
    the last resulting statecount vs. threshold ('laststates-thr', plots only the last result) or time
    vs. threshold ('time-thr'). If no axis is specified, a new subplot is generated.

    :param run: Result of a `run`-call.
    :param mode: Type of plot, defaults to "states-thr"
    :type mode: str, optional
    :param ax: Matplotlib-axis that should be used, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    :param title: Title of plot. If None, the method-description will be used, defaults to None
    :type title: str, optional
    :return: The axis-object that is created or specified in the method-call.
    :rtype: matplotlib.axes.Axes
    """    
    assert mode in ["time-thr", "states-thr", "laststates-thr"]
    if ax is None:
        ax = plt.subplot()
    
    resultcount = len(run["run"][0]["statecounts"])
    if mode in ["states-thr", "laststates-thr"]:
        maxstatecount = max([max(el["statecounts"]) for el in run["run"]])
        normalize = maxstatecount > 10000
        ax.set_ylabel("states (x1000)" if normalize else "states")
        markers = ["o", "x", ".", "v", "x", "^", "d", "s", "*", "h"]
        for idx in range(resultcount):
            if mode == "laststates-thr" and idx != resultcount-1:
                continue
            marker = markers[idx % len(markers)]
            thr = [el["threshold"] for el in run["run"]]
            label = r"%s" % run["method"]["type"] if resultcount == 1 else r"$%s_{%s}$" % (run["method"]["type"], idx+1)
            sta = [el["statecounts"][idx] for el in run["run"]]
            if normalize:
                sta = [el/1000 for el in sta]
            ax.plot(thr, sta, linestyle="dashed", marker=marker, label=label)
    elif mode == "time-thr":
        ax.set_ylabel("time [s]")
        tim = [sum(el["times"]) for el in run["run"]]
        thr = [el["threshold"] for el in run["run"]]
        label = r"%s" % run["method"]["type"]
        ax.plot(thr, tim, linestyle="dashed", marker="x",  label=label)

    ax.set_xlabel(r"threshold $\lambda$")
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return ax