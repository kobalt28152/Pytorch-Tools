import torch
from torch import nn
from torch.nn.functional import logsigmoid, binary_cross_entropy_with_logits

from .functional import IoU, Dice

class IoU_Loss(nn.Module):
    """ IoU loss from logits for binary segmentation

    IoU(a, b) = inter(a, b) / union(a, b) """

    def __init__(self, eps=1e-6):
        """
        Parameters
        ----------
        eps : float
            small value to avoid division by zero and log of zero
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred, y):
        """
        Parameters
        ----------
        pred : torch.Tensor
            predicted mask, shape  (N, 1, H, W)
        y : torch.Tensor
            ground truth mask, shape  (N, 1, H, W)

        Returns
        -------
        torcn.Tensor
            iou score averaged over the batch size
        """
        # IoU can be zero; since log(0) = -inf, clamp iou value to a minimum
        iou = IoU(logsigmoid(pred).exp(), y, self.eps)
        iou = -torch.log(iou.clamp_min(self.eps))

        return iou.mean()


class Dice_Loss(nn.Module):
    """ Dice Loss from logits for binary segmentation.

    Dice(a,b) = 2 * inter(a, b) / (|a| + |b|) """

    def __init__(self, eps=1e-6):
        """
        Parameters
        ----------
        eps : float
            small value to avoid division by zero and log of zero
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred, y):
        """
        Parameters
        ----------
        pred : torch.Tensor
            predicted mask, shape  (N, 1, H, W)
        y : torch.Tensor
            ground truth mask, shape  (N, 1, H, W)

        Returns
        -------
        torcn.Tensor
            dice score averaged over the batch size
        """
        # pred_sig = logsigmoid(pred).exp()

        dice = Dice(logsigmoid(pred).exp(), y, self.eps)
        dice = -torch.log(dice.clamp_min(self.eps))

        return dice.mean()


class MCCLoss(nn.Module):
    """ Matthews Correlation Coefficient (MCC) Loss for image segmentation.

    MCC can only be used for binary segmentation.

    Reference: https://github.com/kakumarabhishek/MCC-Loss """

    def __init__(self, eps: float = 1e-5):
        """
        Parameters
        ----------
            eps : float
                Small epsilon to handle situations where all the samples in the
                dataset belong to one class. """

        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            prediction, shape (N, 1, H, W)
        target : torch.Tensor
            ground truth, shape (N, 1, H, W)

        Returns
        -------
        torch.Tensor:
            loss value (1 - mcc) """

        pred_sig = logsigmoid(pred).exp()
        # bs = y_true.shape[0]

        # y_true = y_true.view(bs, 1, -1)
        # y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(pred_sig, target)) + self.eps
        tn = torch.sum(torch.mul((1 - pred_sig), (1 - target))) + self.eps
        fp = torch.sum(torch.mul(pred_sig, (1 - target))) + self.eps
        fn = torch.sum(torch.mul((1 - pred_sig), target)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        return 1.0 - mcc


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, y):
        with torch.no_grad():
            p = torch.sigmoid(pred)
        pt = p*y + (1-p)*(1-y)

        bce_loss = binary_cross_entropy_with_logits(pred, y, reduction='none')
        loss = bce_loss * ((1-pt)**self.gamma)

        if self.alpha:
            at = self.alpha*y + (1-self.alpha)*(1-y)
            loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()

        return loss   
