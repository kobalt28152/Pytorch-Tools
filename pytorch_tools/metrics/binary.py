import numpy as np
import torch


# ========== Individual metrics ========

def IoU(a, b, eps=1e-6):
    """ IoU score

    Parameters
    ----------
    a : bool
        H x W image
    b : bool
        H x W image

    Returns
    -------
    float
        Iou score between a, b.
    """
    inter = (a*b).sum()
    union = a.sum() + b.sum() - inter

    return inter / (union + 1)    # Add 1 to avoid division by 0

def Dice(a, b, eps=1e-6):
    """ Dice score

    Parameters
    ----------
    a : bool
        H x W image
    b : bool
        H x W image

    Returns
    -------
    float
        Dice score between a, b.
    """
    inter = (a*b).sum()
    card = a.sum() + b.sum()

    return (2.0*inter) / (card + 1)    # Add 1 to avoid division by 0


# ========== Batch metrics ========

class IoU_Metric(object):
    """ IoU metric for binary segmentation.

        IoU(a, b) = inter(a, b) / union(a, b) """

    def __init__(self, p_threshold=0.5, y_threshold=0.01, logits=True):
        """
        Parameters
        ----------
        p_threshold : float
            Prediction threshold
        y_threshold : float
            Target threshold (set almost every non-zero pixel to 1)
        logits : bool
            True if predicted mask comes from logits """
        self.p_threshold = p_threshold
        self.y_threshold = y_threshold
        self.logits = logits
    
    def __call__(self, p, y):
        """
        Parameters
        ----------
        p : torch.tensor
            predicted mask, N x 1 x H x W
        y : torch.tensor
            ground truth mask, N x 1 x H x W

        IoU(a, b) = inter(a,b) / union(a,b), where
            inter(a,b) = a * b
            union(a,b) = |a| + |b| - inter
                       = card - inter,        card = |a| + |b|
                       
        IoU is defined to be 1.0 when both  'a' and 'b' are empty. """
        with torch.no_grad():
            a = torch.sigmoid(p) > self.p_threshold if self.logits else p > self.p_threshold
            b = y > self.y_threshold    # NOTE: a, b are torch.bool
            
            # Consume dimensions 1, 2, 3 => inter is N x 1 tensor
            inter = torch.sum(a * b, dim=(1,2,3))
            card = torch.sum(a, dim=(1,2,3)) + torch.sum(b, dim=(1,2,3))

            ret = inter / (card - inter + 1)    # +1 to avoid division by 0
            ret[card == 0] = 1.0                # when a and b are empty card == 0
            
            return ret.mean()
            
    def __repr__(self):
        return 'IoU_Metric()'


class Dice_Metric(object):
    """ Dice metric for binary segmentation.

        Dice(a,b) = 2 * inter(a, b) / (|a| + |b|) """

    def __init__(self, p_threshold=0.5, y_threshold=0.01, logits=True):
        """
        Parameters
        ----------
        p_threshold : float
            Prediction threshold
        y_threshold : float
            Target threshold (set almost every non-zero pixel to 1)
        logits : bool
            True if predicted mask comes from logits """
        self.p_threshold = p_threshold
        self.y_threshold = y_threshold
        self.logits = logits

    def __call__(self, p, y):
        """
        Parameters
        ----------
        p : torch.tensor
            predicted mask, N x 1 x H x W
        y : torch.tensor
            ground truth mask, N x 1 x H x W

        Dice(a,b) = 2 * inter(a, b) / card(a, b), where 
            inter(a, b) = a * b,
            card(a, b) = |a| + |b|

        The Dice coefficient is defined to be 1.0 when both 'a' and 'b'
        are empty. """
        with torch.no_grad():
            a = torch.sigmoid(p) > self.p_threshold if self.logits else p > self.p_threshold
            b = y > self.y_threshold    # NOTE: a, b are torch.bool
            
            inter = torch.sum(a * b, dim=(1,2,3))
            card = torch.sum(a, dim=(1,2,3)) + torch.sum(b, dim=(1,2,3))
            
            ret = (2.0*inter) / (card + 1)    # +1 to avoid division by zero
            ret[card == 0] = 1.0              # when a and b are empty card == 0
            
            if self.reduction == 'mean': return ret.mean()
            else: return ret
            
    def __repr__(self):
        return 'Dice_Metric()'
