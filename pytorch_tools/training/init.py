import torch
import torch.nn as nn

def init_ICNR(tensor, r=2):
    """ Initialize weights of a Conv2D module to ICNR

    ICNR: Initialized to Convolution NN Resize; i.e. the subset of kernles used in
    the same HR image are initialized to the same value to avoid artifacts.

    For example, consider the following conv operation:

            cin x H x W  -- conv -->  8 x H x W    (Cr^2 x H x W)

    Cr^2 = 8; let  C = 2 and r=2. The kernel producing this tensor must have shape:
    (8 x cin x k x k). If pixel shuffle with r=2 is used, then we get an HR tensor:

            8 x H x W  -- PS -->  2 x 2*H x 2*W
    where
    - channels {0,1,2,3} are used to contruct the 1st channel and
    - channels {4,5,6,7} are used to contruct the 2nd channel

    Then, to avoid artifacts (checkerboard pattern) after initialization, kernel
    weights:
    - {0,1,2,3} are initialized to the same value, and
    - {4,5,6,7} are initialized to the same value (different than the previous)

    Assuming that the weights are already initialized, this function simply takes the
    first C weights from the kernel and repeats them r^2 times (interleaved) such that
    each subset (of size r^2) has the same weights.

    Parameters
    ----------
    tensor : torch.Tensor or nn.Parameter
        kernel weights with shape: cin x cout x k x k
    r : int
        scale factor for pixel shuffle
    """
    cin = tensor.size(0)    # cin == C * r^2
    C = int(cin / (r**2))   # C: channels in HR tensor
    assert C*r**2 == cin    # Make sure that C is in fact a multiple of r^2
    with torch.no_grad():
        # Use the first C elements as base for each subset
        tensor[:] = nn.Parameter(torch.repeat_interleave(tensor[:C], r**2, dim=0))
