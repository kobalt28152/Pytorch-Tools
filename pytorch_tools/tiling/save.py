import numpy as np
import pandas as pd
import cv2

import os

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
    mask : np.ndarray
        image and corresponding mask (H x W x C and H x W x C')

    base_dir : string
        base directory where images should be saved
    
    img_info : pd.Series
        optional pandas series object containing information about the input
        image; e.g. 'id', 'name', etc.

    tile_size : int
    step : int
        tile size and step parameters.

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
