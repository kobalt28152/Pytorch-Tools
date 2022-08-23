import numpy as np
import torch


def augment_Dih4(x, HWC=False):
    """ Augment input tensor by Dihedral Group 4

    That is, given a 3D tensor with shape C x H x W, return a 4D tensor with
    shape 8 x C x H x W:

        id    rot90    rot180    rot270
        AB    BD       DC        CA
        CD    AC       BA        DB

        ref.
        CD    DB       BA        AC
        AB    CA       DC        BD

    Parameters
    ----------
    x : torch.Tensor
        input 3D tensor
    HWC : bool
        format of the input tensor; H x W x C (True) or C x H x W (False)

    Returns
    -------
    torch.Tensor
        4D tensor with augmented images.
    """
    ret = torch.empty((8, *x.shape), dtype=x.dtype)

    dims = (0, 1) if HWC else (1, 2)    # CHW
    fdim = (0,) if HWC else (1,)

    for i in range(4): ret[i] = torch.rot90(x, i, dims=dims)
    x = torch.flip(x, dims=fdim)
    for i in range(4): ret[i+4] = torch.rot90(x, i, dims=dims)

    return ret

def remove_Dih4(x, inplace=False, HWC=False):
    """ Remove DH4 augmentations from the input tensor.

    For each element in the input tensor perform the inverse of the operation
    performed in 'augment_Dih4'.

    Parameters
    ----------
    x : torch.Tensor
        4D tensor containing a batch of images augmented by Dih4.
    inplace : bool
        whether or not operation is perfomed in place. if True, the input
        tensor is modified; else a new tensor is created.
    HWC : bool
        format of the input tensor; 8 x H x W x C (True) or
        8 x C x H x W (False).

    Returns
    -------
    torch.Tensor
        4D tensor containing the modified batch of images.
    """
    ret = x if inplace else torch.empty_like(x)

    dims = (0, 1) if HWC else (1, 2)    # CHW
    fdim = (0,) if HWC else (1,)

    for i in range(4): ret[i] = torch.rot90(x[i], -i, dims=dims)
    for i in range(4): ret[i+4] = torch.flip(torch.rot90(x[i+4], -i, dims=dims), dims=fdim)

    return ret


def batch_transform(batch, transform):
    """ Batch transformation

    Apply 'transform' to each element of 'batch'.

    Parameters
    ----------
    batch : np.ndarray or torch.Tensor
       input batch
            N x -1 (np.ndarray) or
            N x -1 (torch.Tensor)
    transform : function
        transformation function from
            ndarray -> ndarray/Tensor or
            Tensor  -> ndarray/Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
            N x -1 (ndarray/Tensor)
    """
    N = batch.shape[0]
    
    # Compute first element to obtain output dimensions
    tmp = transform(batch[0])
    
    # Given the output dimensions create the output tensor
    if isinstance(tmp, torch.Tensor):
        x = torch.empty((N, *tmp.shape), dtype=tmp.dtype) 
    elif isinstance(tmp, np.ndarray):
        x = np.empty((N, *tmp.shape), dtype=tmp.dtype)
    
    # Populate the output array
    x[0] = tmp
    for i in range(1,N):
        x[i] = transform(batch[i])
    
    return x
