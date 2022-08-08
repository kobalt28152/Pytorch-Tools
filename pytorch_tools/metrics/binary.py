import torch


# ========== Individual metrics ========

def IoU(a, b):
    """ IoU score

    Parameters
    ----------
    a : torch.Tensor
        binary image; N x H x W or H x W
    b : torch.Tensor
        binary image; N x H x W or H x W

    Returns
    -------
    torch.Tensor
        Iou score between a, b; (N, 1) or () tensor
    """
    inter = torch.sum(a*b, dim=(-2,-1))
    card = torch.sum(a, dim=(-2,-1)) + torch.sum(b, dim=(-2,-1))

    # union = a.sum() + b.sum() - inter
    ret = inter / (card - inter + 1)    # Add 1 to avoid division by 0
    ret[card == 0] = 1.0                # when a and b are empty card == 0

    return ret

def Dice(a, b):
    """ Dice score

    Parameters
    ----------
    a : torch.Tensor
        binary image; N x H x W or H x W
    b : torch.Tensor
        binary image; N x H x W or H x W

    Returns
    -------
    torch.Tensor
        Dice score between a, b; (N, 1) or () tensor
    """
    inter = torch.sum(a*b, dim=(-2,-1))
    card = torch.sum(a, dim=(-2,-1)) + torch.sum(b, dim=(-2,-1))
            
    ret = (2.0*inter) / (card + 1)    # +1 to avoid division by zero
    ret[card == 0] = 1.0              # when a and b are empty card == 0

    return ret


# ========== Batch metrics ========

class IoU_Metric(object):
    """ IoU metric for binary segmentation.

        IoU(a, b) = inter(a, b) / union(a, b) """

    def __init__(self, p_threshold=0.5, y_threshold=0.5):
        """
        Parameters
        ----------
        p_threshold : float
            Prediction threshold
        y_threshold : float
            Target threshold (set almost every non-zero pixel to 1)
        """
        self.p_threshold = p_threshold
        self.y_threshold = y_threshold

    def __call__(self, p, y):
        """
        Parameters
        ----------
        p : torch.tensor
            prediction (logits), N x 1 x H x W
        y : torch.tensor
            ground truth mask, N x 1 x H x W

        IoU(a, b) = inter(a,b) / union(a,b), where
            inter(a,b) = a * b
            union(a,b) = |a| + |b| - inter
                       = card - inter,        card = |a| + |b|

        IoU is defined to be 1.0 when both  'a' and 'b' are empty.
        """
        with torch.no_grad():
            a = p.sigmoid() > self.p_threshold
            b = y > self.y_threshold    # NOTE: a, b are torch.bool

            ret = IoU(a, b)    # (N, 1, H, W) -> (N, 1)

            return ret.mean()

    def __repr__(self):
        return 'IoU_Metric()'


class Dice_Metric(object):
    """ Dice metric for binary segmentation.

        Dice(a,b) = 2 * inter(a, b) / (|a| + |b|) """

    def __init__(self, p_threshold=0.5, y_threshold=0.5):
        """
        Parameters
        ----------
        p_threshold : float
            Prediction threshold
        y_threshold : float
            Target threshold (set almost every non-zero pixel to 1)
        """
        self.p_threshold = p_threshold
        self.y_threshold = y_threshold

    def __call__(self, p, y):
        """
        Parameters
        ----------
        p : torch.tensor
            prediction (logits), N x 1 x H x W
        y : torch.tensor
            ground truth mask, N x 1 x H x W

        Dice(a,b) = 2 * inter(a, b) / card(a, b), where 
            inter(a, b) = a * b,
            card(a, b) = |a| + |b|

        The Dice coefficient is defined to be 1.0 when both 'a' and 'b'
        are empty.
        """
        with torch.no_grad():
            a = p.sigmoid() > self.p_threshold    # NOTE: a, b are torch.bool
            b = y > self.y_threshold
            
            ret = Dice(a, b)    # (N, 1, H, W) -> (N, 1)
            
            return ret.mean()
            
    def __repr__(self):
        return 'Dice_Metric()'
