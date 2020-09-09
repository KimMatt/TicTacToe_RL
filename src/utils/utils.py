import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cum_sum(x, discount):
    """
    Computing discounted cumulative sums of vectors.
    """
    discounted_result = np.zeros(len(x), dtype=np.float32)
    discounted_result[-1] = x[-1]
    if len(x) == 1:
        return x
    for i in range(len(x)-2,-1,-1):
        discounted_result[i] = x[i] + discount * discounted_result[i+1]
    return discounted_result


def save_results(win_ratios, loss_ratios, tie_ratios, algo_name, params):
    data = pd.DataFrame({'wins': win_ratios, 'losses': loss_ratios, 'ties': tie_ratios})
    model_name = "_".join(["{}={}".format(param, params[param]) for param in params.keys()])
    title = algo_name + "_" + model_name
    graph = data.plot(kind="line", title=title)
    graph.set_xlabel("epochs")
    f = graph.get_figure()
    y_label = "percentage"
    if y_label:
        graph.set_ylabel(y_label)
    try:
        f.savefig("figs/" + title + ".png")
    except:
        os.mkdir("figs")
        f.savefig("figs/" + title + ".png")
