import numpy as np

def dcg_score(y, k=100):
    """
    Input array must be sorted! \n
    i-th value is a gain from i-th item.
    """
    y = np.array(y)[:k]
    gain = 2 ** y - 1
    discounts = np.log2(np.arange(len(y)) + 2)

    return np.sum(gain / discounts)


def ndcg_score(y_true, y_score, k=100):
    """
    Input arrays must be sorted!
    """
    actual = dcg_score(y_score, k)
    best = dcg_score(y_true, k)
    return actual / best