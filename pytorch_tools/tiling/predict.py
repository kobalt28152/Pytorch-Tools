import numpy as np

import torch
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from .tiling import tile_simple, tile_overlapped, yx_tile_to_pos
from .padding import unpad_img
from ..utils.predict import batch_transform


def predict_fast(img, model, device, n_classes=1, preproc=None, postproc=None,
        tile_size=256, mode='reflect', verbose=0):
    """ Predict full mask by tiling the image with no-overlap.

    Divide 'img' into tiles of shape (tile_size x tile_size),  with no
    overlapping. If required, pad the input image on the (right/bottom). Use
    `model` to make predictions on each generated tile.

    Parameters
    ----------
    img : np.ndarray
        input image (H x W x C)
    model : nn.Model
        NN model 1 x H x W x C -> 1 x H x W x n_classes
    device : torch.device
        cpu or gpu
    preproc : function
    postproc : function
        Preprocessing (before applying model) function and postpocessing (after
        applying model) function.
    tile_size : int
        tile size
    mode : string
        padding mode (reflect, constant)
    verbose : int
        verbosity level: 0 (no message), 1 (basic message),
        2 (basic message + tqdm)

    Returns
    -------
    np.ndarray
        full predicted mask (same size as 'img' H x W x C)
    """
    step = tile_size
    if preproc is None: preproc = lambda x: to_tensor(x)
    if postproc is None: postproc = lambda x: x.squeeze(0).sigmoid().permute(1, 2, 0).numpy()

    # Generate Tiles
    img_tiles, pads = tile_simple(img, tile_size=tile_size, mode=mode)

    height, width = img.shape[:2]
    y_tiles, x_tiles = img_tiles.shape[:2]

    # Predicted mask (padded)
    msk_pred = np.zeros((height+np.sum(pads[0]), width+np.sum(pads[1]), n_classes), dtype=np.float32)


    if verbose > 0:
        print(f'Image divided in ({y_tiles}, {x_tiles}) tiles; processing tiles ({device}) ...')
        if verbose > 1:
            pbar = tqdm(total=y_tiles*x_tiles)

    for j in range(y_tiles):
        for i in range(x_tiles):
            # Pre-process tile
            x = preproc(img_tiles[j,i])

            # Predict mask
            x = x.unsqueeze(0)
            x = x.to(device)
            with torch.no_grad():
                p = model(x)

            p = p.to('cpu')    # Always retur answer to CPU
            p = postproc(p)

            (y0,y1), (x0,x1) = yx_tile_to_pos(j, i, tile_size=tile_size, step=step)
            msk_pred[y0:y1, x0:x1, :] = p

            if verbose > 1:
                pbar.update(1)

    if verbose > 1: pbar.close()
    return unpad_img(msk_pred, pads)


def predict_fast_batch(img, model, device, n_classes=1, preproc=None, postproc=None,
        tile_size=256, mode='reflect', verbose=0):
    """ Predict full mask by tiling the image with no-overlap (batch version)

    Same as predict_fast but process all tiles as a single batch """
    step = tile_size
    if preproc is None: preproc = lambda x: to_tensor(x)
    if postproc is None: postproc = lambda x: x.squeeze(0).sigmoid().permute(1, 2, 0).numpy()

    # Compute tiles
    img_tiles, pads = tile_simple(img, tile_size=tile_size, mode=mode)

    height, width = img.shape[:2]
    y_tiles, x_tiles = img_tiles.shape[:2]
    
    # Predicted mask (padded)
    msk_pred = np.zeros((height+np.sum(pads[0]), width+np.sum(pads[1]), n_classes), dtype=np.float32)

    if verbose > 0:
        print(f'Image divided in ({y_tiles}, {x_tiles}) tiles; processing tiles ({device}) ...')

    # Process all tiles. Flatten the tiles; i.e. reshape
    #       X x Y x H x W x C -> X*Y x H x W x C
    x = batch_transform(img_tiles.reshape(-1, *img_tiles.shape[2:]), preproc)
    x = x.to(device)

    with torch.no_grad():
        p = model(x)
            
    p = p.to('cpu')
    p = batch_transform(p, postproc)

    for j in range(y_tiles):
        for i in range(x_tiles):
            k = j*y_tiles + i    # index in flattened tiles
            (y0,y1), (x0,x1) = yx_tile_to_pos(j, i, tile_size=tile_size, step=step)
            msk_pred[y0:y1, x0:x1, :] = p[k]

    return unpad_img(msk_pred, pads)



def predict_overlapped(img, model, device, n_classes=1, preproc=None, postproc=None,
                             tile_size=256, step=128, mode='reflect', verbose=0):
    """ Predict full mask by tiling the image, where tiles CAN OVERLAP.

    Divide 'img' into tiles of shape (tile_size x tile_size),  with
        overlap. If required, pad the input image on the (left/right) and
        (top/bottom):
        - Using `model` make predictions for each tile.
        - Where several predictions intersect take the average
        - return the averaged prediction

    Parameters
    ----------
    img : np.ndarray
        input image (H x W x C)
    model : nn.Model
        NN model 1 x H x W x C -> 1 x H x W x n_classes
    device : torch.device
        cpu or gpu
    preproc : function
    postproc : function
        Preprocessing (before applying model) function and postpocessing (after
        applying model) function
    tile_size : int
        tile size
    step : int
        step for the tiles
    mode : string
        padding mode (reflect, constant)
    verbose : int
        verbosity level: 0 (no message), 1 (basic message)
                            2 (basic message + tqdm)
    Returns
    -------
    np.ndarray
        full predicted mask (same size as 'img' H x W x C)
    """
    if preproc is None: preproc = lambda x: to_tensor(x)
    if postproc is None: postproc = lambda x: x.squeeze(0).sigmoid().permute(1, 2, 0).numpy()

    # Compute tiles
    img_tiles, pads = tile_overlapped(img, tile_size=tile_size, step=step, mode=mode)

    height, width = img.shape[:2]
    y_tiles, x_tiles = img_tiles.shape[:2]
    
    # Predicted mask and mask for averaging (number of times each pixel is processed)
    msk_pred = np.zeros((height+np.sum(pads[0]), width+np.sum(pads[1]), n_classes), dtype=np.float32)
    avg_msk = np.zeros((height+np.sum(pads[0]), width+np.sum(pads[1])), dtype=np.float32)

    if verbose > 0:
        print(f'Image divided in ({y_tiles}, {x_tiles}) tiles; processing tiles ({device}) ...')
        if verbose > 1:
            pbar = tqdm(total=y_tiles)

    for j in range(y_tiles):
        # Process each row of the tiled image as a batch
        x = batch_transform(img_tiles[j, :], preproc)
        x = x.to(device)

        with torch.no_grad():
            p = model(x)
            
        p = p.to('cpu')
        p = batch_transform(p, postproc)

        for i in range(x_tiles):
            # Add predicted tile to full size mask and increase
            # the number of times pixels are processed
            (y0,y1), (x0,x1) = yx_tile_to_pos(j, i, tile_size=tile_size, step=step)
            msk_pred[y0:y1, x0:x1, :] += p[i]
            avg_msk[y0:y1, x0:x1] += 1

        if verbose > 1:
            pbar.update(1)

    if verbose > 1: pbar.close()

    # Return unpadded full size averaged predicted mask
    # a / b[:,:,None] -> Broadcasting
    return unpad_img(msk_pred/avg_msk[:,:,None], pads)
