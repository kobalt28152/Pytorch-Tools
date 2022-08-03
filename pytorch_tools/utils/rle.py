import numpy as np
from numba import jit

@jit(nopython=True)
def fill_mask(arr, shape):
    """ Actual RLE decoding function (numba compiled)"""
    ret = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for i in range(len(arr)//2):
        px = arr[i*2]-1
        cnt = arr[i*2+1]
        ret[px:px+cnt] = 255
    return ret.reshape(shape).T

def decode_RLE(rle, shape):
    """ Decode RLE (run-lenght encoding)

    Decode RLE string. Format:
            [pos_1 count_1 pos_2 count_2 ...],
    where
        - 'pos_x' is the position; one-indexed and numbered from top to bottom,
          then left to right (counted along the columns).
        - 'count_x' is the number of non-zero elements starting at 'pos_x'.

    Parameters
    ----------
    rle : string
        rle string; e.g. "45 324 1500 13 3000 456 ..."
    shape :  tuple 
        mask shape (height, width) - numpy format (h, w, -1)

    Returns
    -------
    np.ndarray
        binary mask (255 or 0) with shape 'shape' and dtype uint8
    """
    tmp = np.array([int(i) for i in rle.split(' ')])
    return fill_mask(tmp, shape)


@jit(nopython=True)
def fill_rle(mask):
    """ Actual RLE encoding function (numba compiled)"""
    mask = mask.T.ravel()
    idxs = np.where(mask != 0)[0]
    
    rle = [np.int64(x) for x in range(0)]
    if len(idxs) == 0: return [np.int64(x) for x in range(0)]
    
    a = idxs[0]+1    # position
    b = 1            # counter
    
    prev = idxs[0]
    for cur in idxs[1:]:
        if cur == prev+1:
            b += 1
        else:
            rle.extend((a, b))
            a = cur+1
            b = 1
        prev = cur
    else:
        rle.extend((a, b))
        
    return rle

def encode_RLE(mask):
    """ Encode a binar mask into RLE (run-lenght encoding).

    Encode mask into RLE format:
            [pos_1 coun_1 pos_2 count_2 ...].

    Recall that RLE is one-indexed and goes top-bottom then left-right.

    Parameters
    ----------
    np.ndarray
        input mask (H x W) uint8 ndarray

    Returns
    -------
    string
        string with RLE mask
    """
    tmp = fill_rle(mask)
    return ' '.join([str(i) for i in tmp])


# ========== Slow versions =========
# Use pure python, no compiling
#
def decode_RLE_py(rle, shape):
    """ Decode RLE (run-lenght encoding)

    Decode RLE string. Format:
            [pos_1 count_1 pos_2 count_2 ...],
    where
        - 'pos_x' is the position; one-indexed and numbered from top to bottom,
          then left to right (counted along the columns).
        - 'count_x' is the number of non-zero elements starting at 'pos_x'.

    Parameters
    ----------
    rle : string
        rle string; e.g. "45 324 1500 13 3000 456 ..."
    shape :  tuple 
        mask shape (height, width) - numpy format (h, w, -1)

    Returns
    -------
    np.ndarray
        binary mask (255 or 0) with shape 'shape' and dtype uint8
    """
    arr = np.array([int(i) for i in rle.split(' ')])
    ret = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for i in range(len(arr)//2):
        px = arr[i*2]-1
        cnt = arr[i*2+1]
        ret[px:px+cnt] = 255

    return ret.reshape(shape).T

def encode_RLE_py(mask):
    """ Encode a binar mask into RLE (run-lenght encoding).

    Encode mask into RLE format:
            [pos_1 coun_1 pos_2 count_2 ...].

    Recall that RLE is one-indexed and goes top-bottom then left-right.

    Parameters
    ----------
    np.ndarray
        input mask (H x W) uint8 ndarray

    Returns
    -------
    string
        string with RLE mask
    """
    mask = mask.T.ravel()
    idxs = np.where(mask != 0)[0]
    
    rle = [np.int64(x) for x in range(0)]
    if len(idxs) == 0: return [np.int64(x) for x in range(0)]
    
    a = idxs[0]+1    # position
    b = 1            # counter
    
    prev = idxs[0]
    for cur in idxs[1:]:
        if cur == prev+1:
            b += 1
        else:
            rle.extend((a, b))
            a = cur+1
            b = 1
        prev = cur
    else:
        rle.extend((a, b))
        
    return ' '.join([str(i) for i in rle])
