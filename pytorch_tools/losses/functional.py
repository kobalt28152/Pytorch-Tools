import torch

def IoU(a, b, eps=1e-6):
    """ IoU score

    Given two tensors with shape (N, C, H, W) compute their IoU score for
    each class, for each batch element.
        IoU(a,b) = inter(a, b) / union(a, b), where
        union(a, b) = |a| + |b| - inter(a, b)

    Parameters
    ----------
    a : torch.Tensor
        input tensor; (N, C, H, W)
    b : torch.Tensor
        input tensor; (N, C, H, W)
    eps : float
        small number to avoid division by zero

    Returns
    -------
    torch.Tensor
        IoU score for each class, for each batch element; (N, C)
    """
    inter = torch.sum(a * b, dim=(-2,-1))
    union = torch.sum(a, dim=(-2,-1)) + torch.sum(b, dim=(-2,-1)) - inter

    return inter / union.clamp_min(eps)

def Dice(a, b, eps=1e-6):
    """ Dice score

    Given two tensors with shape (N, C, H, W) compute their Dice score for
    each class, for each batch element.
        Dice(a,b) = inter(a, b) / card(a, b), where
        card(a, b) = |a| + |b|

    Parameters
    ----------
    a : torch.Tensor
        input tensor; (N, C, H, W)
    b : torch.Tensor
        input tensor; (N, C, H, W)
    eps : float
        small number to avoid division by zero

    Returns
    -------
    torch.Tensor
        Dice score for each class, for each batch element; (N, C)
    """
    inter = torch.sum(a * b, dim=(-2,-1))
    card = torch.sum(a, dim=(-2,-1)) + torch.sum(b, dim=(-2,-1))

    return (2.0*inter) / card.clamp_min(eps)
