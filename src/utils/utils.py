import numpy as np
import scipy.signal

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
