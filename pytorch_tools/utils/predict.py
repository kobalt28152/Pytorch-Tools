import numpy as np
import torch

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
