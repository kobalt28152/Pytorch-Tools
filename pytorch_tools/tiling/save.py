import numpy as np
import pandas as pd
import cv2

import os
import uuid

from .tiling import tile_imgMsk

def save_tiles(img, mask, base_dir, img_info=None, tile_size=512, step=256,
        out_size=256, reject=None):
    """ Write image tiles to disk.

    Given an image and its corresponding mask ('img', 'msk') create tiles of
    size 'tile_size' by taking steps of size 'step' in the image and the mask.
    Reject empty images (blank images) with absolutely no mask in them by
    considering the mean and standard deviation of the tile.
        
    Save each tile and its corresponding mask in the directories:
            
        'base_dir/train_tiles_images' and 'base_dir/train_tiles_masks'
            
    If 'img_info' is provided it uses the 'id' column to set the base name of
    the tiled images. If not provided a default name is used.

    If 'out_size' != 'tile_size', resize the tiled image and mask before
    saving.

    Parameters
    ----------
    img : np.ndarray
        input image (H x W x C)
    mask : np.ndarray
        corresponding mask (H x W x C')
    base_dir : string
        base directory where images should be saved
    img_info : pd.Series
        optional pandas series object containing information about the input
        image; e.g. 'id', 'name', etc.
    tile_size : int
        tile size
    step : int
        tile step
    out_size : int
        output tile size
        
    Returns
    -------
    List[pd.Series]
        Array containing tile information; each row contains 'img_info` with
        two additional columns: 'img_path', 'msk_path' (the relative paths for
        each tiled image and mask).
    """
    if img_info is None:
        img_info = pd.Series(dtype=object)
        img_info['id'] = 0

    # Save directories
    img_path = os.path.join(base_dir, 'train_tiles_images')
    msk_path = os.path.join(base_dir, 'train_tiles_masks')

    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(msk_path): os.makedirs(msk_path)

    # Tile image
    img_tiles, msk_tiles = tile_imgMsk(img, mask, tile_size=tile_size, step=step)
    y_steps, x_steps = img_tiles.shape[:2]
    img_id = img_info['id']

    # Save each tile (if not emtpy)
    arr = []
    for y in range(y_steps):
        for x in range(x_steps):
            name = f'{img_id}_{x}-{y}.png'
            tile_img = img_tiles[y, x]
            tile_msk = msk_tiles[y, x]

            if out_size != tile_size:
                # Use inter area since images are always downsampled
                tile_img = cv2.resize(tile_img, (out_size, out_size), interpolation=cv2.INTER_AREA)
                tile_msk = cv2.resize(tile_msk, (out_size, out_size), interpolation=cv2.INTER_AREA)

            # If reject function provided:
            if reject is not None:
                # Only discard parts with no mask in them.
                empty = tile_msk.sum() == 0    # empty mask
                if empty and reject(tile_img):
                    continue

            # Save image and mask
            fname_img = os.path.join(img_path, name)
            fname_msk = os.path.join(msk_path, name)
            tf0 = cv2.imwrite(fname_img, tile_img)
            tf1 = cv2.imwrite(fname_msk, tile_msk)

            if not tf0 or not tf1:
                raise IOError(f'Unable to write: {fname_img} and {fname_msk}')

            # Tile information
            info = img_info.copy()
            info['img_path'] = fname_img
            info['msk_path'] = fname_msk
            arr.append(info)

    return arr


def save_tiles_npz(img, msk, save_dir='', img_info=pd.Series(dtype=object),
        tile_size=512, step=256, reject=None, transform=None, verbose=False):
    """ Write tiles to disk.

    Parameters
    ----------
    img : np.ndarray
        input image (H x W x C)
    mask : np.ndarray
        corresponding mask (H x W x C')
    save_dir : string
        directory where tiles are saved; if does not exists, create it.
    img_info : pd.Series
        optional pandas series object containing information about the input
        image; e.g. 'name', 'state', 'zip', 'class' etc.
        If img_info has an 'id' column it will be used as the name for the
        output file. Otherwise a random unique 'id' is created.
    tile_size : int
        tile size
    step : int
        tile step
    reject : function: img -> bool
        if provided: function applied to each tile (only image).
        If reject(tile[i]) == True, tile[i] won't be included in the output.
    transform : function: image, mask -> image, mask
        transformation function applied to the tiles (image and mask).
        Intended use: resize tiles.
    verbose : bool
        verbosity

    Returns
    -------
    pd.Series or None
        img_info augmented with information about the saved tiles:
        {number of tiles, height, width, tiles path}.
        Or None if all tiles where rejected by 'reject'.
    """
    # if save directory does not exists create it
    if save_dir != '' and not os.path.exists(save_dir): os.makedirs(save_dir)

    # Create unique 'id' if no 'id' provided
    if not 'id' in img_info:
        while True:
            name = str(uuid.uuid4())
            if not os.path.exists(os.path.join(save_dir, f'{name}.npz')): break
            
        if verbose: print(f"'id' not in img_info, random 'id' generated: {name}")
        img_info['id'] = name

    # Compute Tiles: Y x X x H x W x C (5D ndarray)
    img_tiles, msk_tiles = tile_imgMsk(img, msk, tile_size=tile_size, step=step)

    # Reshape: Y*X x H x W x C (4D ndarray)
    img_tiles = img_tiles.reshape((-1, *img_tiles.shape[2:]))
    msk_tiles = msk_tiles.reshape((-1, *msk_tiles.shape[2:]))

    if verbose: print(f'Total number of tiles: {len(img_tiles)}')

    # Reject tiles that do not meet the criteria established by 'reject(img)'
    if reject:
        if verbose: print(f'reject ... ', end='', flush=True)

        rejected = np.zeros(len(img_tiles), dtype=bool)
        for i in range(len(img_tiles)):
            rejected[i] = reject(img_tiles[i])

        img_tiles = img_tiles[~rejected]
        msk_tiles = msk_tiles[~rejected]

        if verbose: print(f'{rejected.sum()} rejected (total tiles: {len(img_tiles)})')
        # every tile is rejected: finish execution
        if img_tiles.shape[0] == 0: return None

    # Transforms tiles (image and mask) according to the function 'transform(img, msk)'
    if transform:
        if verbose: print(f'transform ... ', end='', flush=True)

        # Transform the first element to obtain the output shape
        im, mk = transform(img_tiles[0], msk_tiles[0])
        img_trans = np.empty((img_tiles.shape[0], *im.shape), dtype=im.dtype)
        msk_trans = np.empty((msk_tiles.shape[0], *mk.shape), dtype=mk.dtype)

        img_trans[0] = im
        msk_trans[0] = mk

        for i in range(1, len(img_tiles)):
            im, mk = transform(img_tiles[i], msk_tiles[i])
            img_trans[i] = im
            msk_trans[i] = mk

        img_tiles = img_trans
        msk_tiles = msk_trans

        if verbose: print(f'new shapes: {im.shape}, {mk.shape}')

    tiles_path = os.path.join(save_dir, f'{img_info["id"]}')
    np.savez_compressed(tiles_path, image=img_tiles, mask=msk_tiles)
    if verbose: print(f'tiles saved as: {tiles_path}.npz')

    # Generate output info
    info = img_info.copy()
    info['num_tiles'] = len(img_tiles)
    info['height'] = img_tiles.shape[1]
    info['width'] = img_tiles.shape[2]
    info['img_C'] = img_tiles.shape[3]
    info['msk_C'] = msk_tiles.shape[3]
    info['tiles_path'] = f'{tiles_path}.npz'

    return info
