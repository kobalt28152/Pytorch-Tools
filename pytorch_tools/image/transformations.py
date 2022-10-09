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

def collapse_multiLabeled(x, thr, logits=True):
    """ Colapse multi-labeled to multi-class

    Given a multi-labeled tensor (C, H, W) or (N,C,H,W), compute the class for
    each pixel as follows:
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

def increase_equal(a, d):
    """ Increase a range on both sides.

    Increase a range on both sides by an equal amount; or almost equal if
    distance 'd' is not even. That is, suppose that the distance is even, then
        
        [a0, a1) -> [a0+d/2, a1+d/2)

    Since this function is intended for ranges in images, a0 cannot be negative.
    Parameters
    ----------
    a : tuple(2)
        range [a0, a1)
    d : int
        distance to increase
    Returns
    -------
    tuple(2)
        new range [a0+d/2, a1+d/2)
    """
    a0, a1 = a
    td = a1-a0+d    # Total distance
    a0 = max(a0 - d//2, 0)
    a1 = a0 + td
    return (a0, a1)

def match_range_lengths(a, b):
    """ Match two range lengths.

    Match the length of two ranges to the largest length by adding an equal
    quantity on both sides of the smalles range. For example, let

        a: [20, 30), b: [40, 60), 
        => len(a)=10, len(b)=20; so max(len(a), len(b)) = 20
        => must add 20 - len(a) = 10 on both sides of a to match both ranges
        => new ranges are:
        a: [15, 35), b: [40, 60)

    Parameters
    ----------
    a : tuple(2)
        first range [a0, a1)
    b : tuple(2)
        second range [b0, b1)

    Returns
    -------
    tuple(2)
        new first range
    tuple(2)
        new second range
    """
    len_a = a[1]-a[0]
    len_b = b[1]-b[0]
    r = max(len_a, len_b)
    a = increase_equal(a, r-len_a)    # For either (a,b), r-len_{a|b} will be
    b = increase_equal(b, r-len_b)    # zero and won't be modified.

    return a, b
