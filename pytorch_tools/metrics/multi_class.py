import torch

# one_hot takes LongTensor with index values of shape (*) and returns
# a tensor of shape (*, num_classes); e.g. (128x128) -> (128 x 128 x 4)
from torch.nn.functional import one_hot

def IoU(a, b):
    """ IoU Multi-class

    Compute IoU score for each class; i.e. for inputs (a, b), with shape
    H x W x C, compute
        iou[i] = inter(a[...,i], b[...,i]) / union(a[...,i], b[...,i])
    for i in {0, 1, ..., C-1}

    Parameters
    ----------
    a : torch.Tensor
        one-hot encoded labels; N x H x W x C or H x W x C
    b : torch.Tensor
        one-hot encoded labels; N x H x W x C or H x W x C

    Returns
    -------
    torch.Tensor
        IoU score for each class; N x C or C
    """
    inter = torch.sum(a * b, dim=(-3,-2))
    card = torch.sum(a, dim=(-3,-2)) + torch.sum(b, dim=(-3,-2))

    ret = inter / (card - inter + 1)    # + 1 to avoid division by 0
    ret[card == 0] = 1.0    # Set to 1.0, where: |a| == 0 and |b| == 0

    return ret


def Dice(a, b):
    """ Dice Score Multi-class

    Compute Dice score for each class; i.e. for inputs (a, b), with shape
    H x W x C, compute
        dice[i] = 2*inter(a[...,i], b[...,i]) / (|a[...,i]| + |b[...,i]|)
    for i in {0, 1, ..., C-1}

    Parameters
    ----------
    a : torch.Tensor
        one-hot encoded labels; N x H x W x C or H x W x C
    b : torch.Tensor
        one-hot encoded labels; N x H x W x C or H x W x C

    Returns
    -------
    torch.Tensor
        Dice score for each class; N x C or C
    """
    # (N, H, W, C) -> (N, C)    or    (H, W, C) -> (C,)
    inter = torch.sum(a * b, dim=(-3, -2))    
    card = torch.sum(a, dim=(-3,-2)) + torch.sum(b, dim=(-3,-2))

    ret = (2.0*inter) / (card + 1)    # +1 to avoid division by zero
    ret[card == 0] = 1.0              # when a and b are empty card == 0

    return ret


class IoU_Metric(object):
    """ IoU metric for multi-class segmentation.

    For each class, compute the IoU beween corresponding elements. That is, for
    a (N x num_classes x H x W) prediction and target. Compute the IoU:

    - for each element in the batch (j)
    - for each class in num_classes (i)

        a = j x i x H x W
        b = j x i x H x W

        IoU(a, b) = inter(a, b) / union(a, b)

    This generates an N x num_classes tensor with the IoU scores for each class
    in each batch element. This tensor is then reduced to either:

    - num_classes x 1, tensor containing the average for each class
    - scalar (tensor), containing the average between all clases.
    """

    def __init__(self, num_classes, per_class=True):
        """
        Parameters
        ----------
        num_classes : int
            number of classes
        per_class : bool
            compute IoU per class (True -> return vector; False -> Scalar)
        """
        self.num_classes = num_classes
        self.per_class = per_class

    def __call__(self, pred, y):
        """
        Parameters
        ----------
        pred : torch.Tensor
            logits (predicted values): N x num_classes x H x W (float32)
        y : torch.Tensor
            target (labels): N x H x W (long)
        """
        with torch.no_grad():
            # x :                      N x C x H x W    (logits)
            # x = softmax(x) :         N x C x H x W    (probabilities)
            # x = argmax(x, dim=1) :   N x H x W        (labels)
            # x = one_hot(x, C)) :     N x H x W x C    (one-hot encoded)
            p_hot = one_hot(pred.softmax(dim=-3).argmax(dim=-3), num_classes=self.num_classes)
            y_hot = one_hot(y, num_classes=self.num_classes)

            ret = IoU(p_hot, y_hot)    # N x C
            
            if self.per_class: return torch.mean(ret, dim=0)    # (C,) tensor
            else: return torch.mean(ret)    # 1 x 1 (actually Size[]; i.e. no dims)
        
    def __repr__(self):
        return 'IoU_Metric()'


class Dice_Metric(object):
    """ Dice metric for multi-class segmentation.

    For each class, compute the Dice score beween corresponding elements. That
    is, for a (N x num_classes x H x W) prediction and target. Compute the
    Dice:

    - for each element in the batch (j)
    - for each class in num_classes (i)

        a = j x i x H x W
        b = j x i x H x W

        Dice(a, b) = 2.0*inter(a, b) / (|a| + |b|)

    This generates a (N x num_classes) tensor with the Dice scores for each
    class in each batch element. This tensor is then reduced to either:

    - num_classes x 1, tensor containing the average for each class
    - scalar (tensor), containing the average between all clases.
    """

    def __init__(self, num_classes, per_class=True):
        """
        Parameters
        ----------
        num_classes : int
            number of classes
        per_class : bool
            compute IoU per class (True -> return vector; False -> Scalar)
        """
        self.num_classes = num_classes
        self.per_class = per_class

    def __call__(self, pred, y):
        """
        Parameters
        ----------
        pred : torch.Tensor
            logits (predicted values): N x num_classes x H x W (float32)
        y : torch.Tensor
            target (labels): N x H x W (long)
        """
        with torch.no_grad():
            # x :                      N x C x H x W    (logits)
            # x = softmax(x) :         N x C x H x W    (probabilities)
            # x = argmax(x, dim=1) :   N x H x W        (labels)
            # x = one_hot(x, C)) :     N x H x W x C    (one-hot encoded)
            # p_hot = one_hot(torch.argmax(self.softmax(pred), dim=1), num_classes=self.num_classes)
            p_hot = one_hot(pred.softmax(dim=-3).argmax(dim=-3), num_classes=self.num_classes)
            y_hot = one_hot(y, num_classes=self.num_classes)

            ret = Dice(p_hot, y_hot)    # N x H x W x C -> N x C

            if self.per_class: return torch.mean(ret, dim=0)    # (C,) tensor
            else: return torch.mean(ret)    # scalar tensor (actually Size[]; i.e. no dims)

    def __repr__(self):
        return 'Dice_Metric()'

