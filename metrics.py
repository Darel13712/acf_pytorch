import numpy as np
import pandas as pd

def dcg_score(y, k=100):
    y = np.array(y)[:k]
    gain = 2 ** y - 1
    discounts = np.log2(np.arange(len(y)) + 2)

    return np.sum(gain / discounts)


def ndcg_score(y_true, y_score, k=100):
    actual = dcg_score(y_score, k)
    best = dcg_score(y_true, k)
    return actual / best