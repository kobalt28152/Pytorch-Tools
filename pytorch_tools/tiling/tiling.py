import numpy as np
from .padding import pad_rb, pad_lrtb

def yx_tile_to_pos(y, x, tile_size=256, step=128):
    """ Range from tile position
    Given 'tile_size' and 'step', compute the range
            [row:row+tile_size, col:col+tile_size] 

    Parameters
    ----------
    y : int
    x : int
        Position in the tile (row, col)
    tile_size : int
    step : int
        tile size and step

    Returns
    -------
    tuple
        (y0, y1): range of rows covered by tile 'y' [y0:y1]
    tuple
        (x0, x1): range of columns covered by tile 'x' [x0:x1]
    """
    return (y*step,y*step+tile_size), (x*step,x*step+tile_size)


def tile_simple(img, tile_size=256, mode='reflect'):
    """ Simple Tiling

    Tile the input into tiles of size (tile_size x tile_size). Use simple
    padding (right/bottom) to create an image whose (height, width) are
    multiples of tile_size, then create (y_tiles x x_tiles) tiles.

    Parameters
    ----------
    img : np.ndarray
        input image (H x W x C) image
    tile_size : int
        tile size
    mode : string
        padding mode

    Returns
    -------
    np.ndarray
        5D array containing tiles (y_tiles x x_tiles x H x W x C)
    list[tuple]
        paddin used on the input image: [(0, y_pad), (0, x_pad)]
    """
    C = img.shape[2]
    img_padded, (y_tiles, x_tiles), pads = pad_rb(img, tile_size=tile_size, mode=mode)
    img_tile = np.empty((y_tiles, x_tiles, tile_size, tile_size, C), dtype=np.uint8)
    
    step = tile_size
    for j in range(y_tiles):
        py = j * step
        for i in range(x_tiles):
            px = i * step
            img_tile[j, i] = img_padded[py:py+step, px:px+step, :]

    return img_tile, pads


def tile_overlapped(img, tile_size=256, step=128, mode='reflect'):
    """ Overlapped Tiling

    Tile the input into tiles of size (tile_size x tile_size). Pad the input
    image on all edges (left, right, top, bottom) to create an image where the
    tiles fit exactly (considering the step).

    Parameters
    ----------
    img : np.ndarray
        input image (H x W x C)

    Returns
    -------
    np.ndarray
        5D array containing tiles (y_tiles x x_tiles x H x W x C)
    list[tuple]
        padding used: [(y_pad0, y_pad1), (x_pad0, x_pad1)] """
    C = img.shape[2]
    img_padded, (y_tiles, x_tiles), pads = pad_lrtb(img, tile_size=tile_size, step=step, mode=mode)
    img_tile = np.empty((y_tiles, x_tiles, tile_size, tile_size, C), dtype=np.uint8)
    
    for j in range(y_tiles):
        py = j * step
        for i in range(x_tiles):
            px = i * step
            img_tile[j, i] = img_padded[py:py+tile_size, px:px+tile_size, :]

    return img_tile, pads


def tile_imgMsk(img, mask, tile_size=512, step=256, verbose=False):
    """ Tile an image and its mask such that no padding is added.

    Compute tiles from an image and its mask given the 'tile_size' and the
    'step' such that NO PIXEL is left uncovered and no border is added. If
    there are pixels left on an axis when tiling is performed, simply add a
    tile at the end of the axis; e.g., for size=10, tile=3 and step=3, exactly
    3 tiles can be formed: |x--x--x--0| but there is 1 pixel left uncovred. To
    cover this pixel, simply add a new tile at the end: |x--x--xx--|, where
        x : initial position of tile
        - : covered pixel
        0 : uncovered pixel

    The purpose of this function is to create TRAINING images and masks.

    Parameters
    ----------
    img : np.ndarray
        input image; 3D array (H x W x C)
    mask : np.ndarray
        input mask; 3D array (H x W x C')
    tile_size : int
        tile size
    step : int
         step for tiling
    verbose : bool
        whether to display messages or not

    Returns
    -------
    np.ndarray
        tiled image; 5D array (X x Y x H x W x C)
    np.ndarray
        tiled image; 5D array (X x Y x H x W x C')
    """

    
    height, width = img.shape[:2]
    
    x_steps = int(np.ceil((width - tile_size + 1)/step))
    y_steps = int(np.ceil((height - tile_size + 1)/step))
    
    # Remaining pixels
    tmp_x = x_steps + 1 if width - x_steps*step > 0 else x_steps
    tmp_y = y_steps + 1 if height - y_steps*step > 0 else y_steps
    
    if verbose:
        print(f'Remaining pixels: ({width-x_steps*step}, {height-y_steps*step})')
        print(f'tiles: {tmp_x}, {tmp_y}')
        
    C_img = img.shape[2]
    C_msk = mask.shape[2]

    imgs = np.empty((tmp_y, tmp_x, tile_size, tile_size, C_img), dtype=np.uint8)
    msks = np.empty((tmp_y, tmp_x, tile_size, tile_size, C_msk), dtype=np.uint8)
    for y in range(tmp_y):
        py = height-tile_size if (tmp_y != y_steps and y == tmp_y-1) else y*step
        for x in range(tmp_x):
            px = width-tile_size if (tmp_x != x_steps and x == tmp_x-1) else x*step
            # Get tile from image and mask
            imgs[y, x] = img[py:py+tile_size, px:px+tile_size]
            msks[y, x] = mask[py:py+tile_size, px:px+tile_size]
            
    return imgs, msks
