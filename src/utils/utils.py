import torch
import scipy.signal

def cum_sum(x, discount):
    """
    Computing discounted cumulative sums of vectors.
    """
    discounted_result = torch.zeros(len(x), dtype=torch.float)
    discounted_result[-1] = x[-1]
    for i in range(len(x)-2,-1,-1):
        discounted_result[i] = discount * discounted_result[i+1]
    return discounted_result
