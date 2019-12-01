import torch


def bpr_loss(pos, neg, b=0.0, collapse=True):
    """
    Usual BPR loss that penalizes when negative score is greater than positive one.

    Parameters
    ----------
    pos: torch.tensor
        positive score

    neg: torch.tensor
        negative scores. Can be a batch of negatives.

    b: float
        desired boundary between positive and negative examples

    collapse: bool
        If True collapse batch with mean.

    Returns
    -------
    torch.tensor
    """

    res = torch.sigmoid(neg - pos + b)
    if collapse:
        res = res.mean()
    return res


def warp_loss(pos, neg, b=1, collapse=True):
    """
    Batch version of WARP loss.

    Regular version samples one negative example until violation is met.
    This number of samples become the estimation of rank and weight is produces as some function of this rank.

    This version uses a batch of negatives and estimates rank as a number of violated examples.

    If you use the number of first violation as rank this will degenerate to usual WARP with a limit on draws.

    Parameters
    ----------
    pos: torch.tensor
        positive score

    neg: torch.tensor
        negative scores. Can be a batch of negatives.

    b: float
        desired boundary between positive and negative examples

    collapse: bool
        If True collapse batch with mean.

    Returns
    -------
    torch.tensor
    """

    loss = bpr_loss(pos, neg, b, collapse=False)
    m = (loss > 0.5).float()
    m *= torch.log(m.sum() + 1) + 1
    res = m * loss
    if collapse:
        res = res.mean()
    return res


def ewarp_loss(pos, pos_score, neg, neg_score, b=1, max_rating=5.0, collapse=True):
    """
    Explicit version of WARP loss.

    This takes additional inputs of true positive and negative scores to weight loss by
    (positive + max - negative) / max

    Parameters
    ----------
    pos: torch.tensor
        positive score

    pos_score: float
        explicit positive score

    neg: torch.tensor
        negative scores. Can be a batch of negatives.

    neg_score: [floats]
        explicit negative scores for each item from batch

    b: float
        desired boundary between positive and negative examples

    max_rating: float
        maximum possible value of rating in the dataset

    collapse: bool
        If True collapse batch with mean.

    Returns
    -------
    torch.tensor
    """
    p = torch.tensor([pos_score])
    n = torch.tensor(neg_score, dtype=torch.float)
    m = torch.tensor([max_rating])
    p = torch.repeat_interleave(p, len(n))
    m = torch.repeat_interleave(m, len(n))
    res = ((p + m - n) / m * warp_loss(pos, neg, b, collapse=False))
    if collapse:
        res = res.mean()
    return res

