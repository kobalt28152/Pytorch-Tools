import numpy as np
import torch

import cv2

def resize_keepAspectRatio(img, target=512, value=0, center=True, interpolation=cv2.INTER_LINEAR):
    """ Resize image to (target, target) while keeping aspect ratio

    For example, if input image is (768, 512) and target=512, then resized
    image will be of size (512, 341), remaining pixels will then be filled with
    value to generate a (512, 512) image.

    Parameters
    ----------
    img : np.ndarray
        input image; H x W x C or (H x W)

    target : int
        target size, always squared: height=target, width=target.

    value : int or float
        background value

    center : bool
        whether or not to center the resized image (only useful when one of
        the sides of the resized image is smaller than target size).

    interpolation : int (cv2.INTER_xxx)
        interpolation used in image resize

    Returns
    -------
    np.ndarray
        resized image
    """
    height, width = img.shape[:2]    # img: H x W x C (or H x W)

    r = min(target/height, target/width)
    h = round(r*height)
    w = round(r*width)

    tmp = cv2.resize(img, (w,h), interpolation=interpolation)

    if img.ndim == 3:
        ret = value * np.ones((target, target, img.shape[2]), dtype=img.dtype)
    else:
        ret = value * np.ones((target, target), dtype=img.dtype)

    if center:
        pos_y = int(max(target - h, 0)/2)
        pos_x = int(max(target - w, 0)/2)
        ret[pos_y:pos_y+h, pos_x:pos_x+w] = tmp
    else:
        ret[:h, :w] = tmp

    return ret


def onehot_to_labeled(x, hwc=True):
    """ One-hot encoded mask to Labeled mask

    NOTE: x is expected to be a one-hot encoded tensor; i.e. for each pixel
    only one class is set to 1 and all other classes are 0. If this is not the
    case, this function will fail to produce a meaningful answer.

    Parameters
    ----------
    x : torch.Tensor
        input mask.
    hwc : bool
        shape of input tensor, either
            (N, C, H, W) or (C, H, W); or
            (N, H, W, C) or (H, W, C); or

    Returns
    -------
    torch.Tensor
        labeled mask
    """
    if hwc:
        n = x.size(-1)
        return torch.sum(x * torch.arange(1, n+1), dim=-1)
    else:
        n = x.size(-3)
        return torch.sum(x * torch.arange(1, n+1).reshape(n, 1, 1), dim=-3)

def colapse_multiLabeled(x, thr, logits=True):
    """ Colapse multi-labeled to multi-class

    Given a multi-labeled tensor (C, H, W), compute the class for each pixel as
    follows:
    - Threshold each class using 'thr'
    - For each pixel, where all classes are False, set the pixel as background.
    - Set each pixel to the class with the highest probability.
    - Where a pixel is set as background, set the pixel to background

    Parameters
    ----------
    x : torch.Tensor
        input tensor
    thr : float of torch.Tensor
        threshold for each class
    logits : bool
        if True, x comes from logits; else x is a probability.

    Returns
    -------
    torch.Tensor
        multi-class output (N, H, W) or (H, W) - torch.int64
    """
    pr = x.sigmoid() if logits else x
    
    background = torch.sum(pr > thr, dim=-3) == 0    # N x H x W
    labels = pr.argmax(dim=-3)+1    # N x H x W
    labels[background] = 0          # N x H x W
    
    return labels
