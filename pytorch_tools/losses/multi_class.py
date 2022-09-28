import torch
import torch.nn as nn
from torch.nn.functional import one_hot, logsigmoid

from .functional import IoU, Dice

class IoU_Loss(nn.Module):
    """ IoU loss from logitis

    IoU for multi-class or multi-label segmentation.

    - Multi-class: A single output class is expected. That is each pixel can
      belong to only one class. The output of the network is a Softmax
      function.

    - Multi-label: Multiple output classes are expected. That is, each pixel
      can belong to multiple classes. The output of the network is a Sigmoid
      function.

    Expected shapes:

    - Multi-class:
        - pred : N x C x H x W
        - target: N x H x W    (labeled mask)

    - Multi-label:
        - pred : N x C x H x W
        - target: N x C x H x W    (binary masks)

    For each batch and for each class we compute:

        IoU(a, b) = inter(a, b) / union(a, b)
    """
    def __init__(self, multi_class=True, num_classes=None, eps=1e-6):
        """
        Parameters
        ----------
        multi_class : bool
            if True: multi-class problem expected.
            if False: multi-label problem expected.
        num_classes : int
            number of classes. Only used when multi_class == True.
        eps : float
            small value
        """
        super().__init__()
        self.eps = eps
        self.multi_class = multi_class
        self.num_classes = num_classes

    def forward(self, p, y):
        if self.multi_class:
            p_prob = p.log_softmax(dim=1).exp()     # N x C x H x W (probabilities)
            y_hot = one_hot(y, self.num_classes)    # N x H x W x C (one-hot)
            y_hot = y_hot.type(p_soft.dtype).permute(0, 3, 1, 2)    # N x C x H x W
        else:
            p_prob = logsigmoid(p).exp()    # N x C x H x W (probabilities)
            y_hot = y                       # N x C x H x W (binary targets)

        iou = IoU(p_prob, y_hot, eps=self.eps)        # N x C
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
