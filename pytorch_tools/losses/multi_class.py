import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from .functional import IoU, Dice

class IoU_Loss(nn.Module):
    """ IoU loss from logitis for multiclass segmentation.

    For each batch and for each class compute:

        IoU(a, b) = inter(a, b) / union(a, b)
    """

    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, p, y):
        p_soft = p.log_softmax(dim=1).exp()     # N x C x H x W (probabilities)
        y_hot = one_hot(y, self.num_classes)    # N x H x W x C (one-hot)
        y_hot = y_hot.type(p_soft.dtype).permute(0, 3, 1, 2)    # N x C x H x W

        iou = IoU(p_soft, y_hot, eps=self.eps)        # N x C
        iou = - torch.log(iou.clamp_min(self.eps))    # clamp to eps to avoid log(0)

        return iou.mean()



class Dice_Loss(nn.Module):
    """ Dice loss from logitis for multiclass segmentation.

    For each batch and for each class compute:

        Dice(a, b) = inter(a, b) / (|a| + |b|)
    """

    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, p, y):
        p_soft = p.log_softmax(dim=1).exp()     # N x C x H x W (probabilities)
        y_hot = one_hot(y, self.num_classes)    # N x H x W x C (one-hot)
        y_hot = y_hot.type(p_soft.dtype).permute(0, 3, 1, 2)    # N x C x H x W

        dice = Dice(p_soft, y_hot, eps=self.eps)        # N x C
        dice = - torch.log(dice.clamp_min(self.eps))    # clamp to eps to avoid log(0)

        return dice.mean()
