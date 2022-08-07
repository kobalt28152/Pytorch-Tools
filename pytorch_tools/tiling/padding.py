import numpy as np

def pad_rb(img, tile_size=256, mode='reflect'):
    """ Pad right/bottom.

    Padd an image on the (bottom, right) edges: if (height, width) are not a
    multiple of 'tile_size', pad the input image on the (bottom, right).

    Parameters
    ----------
    img : np.ndarray
        input image (H x W x C)
    tile_size : int
        tile size
    mode : string
        pading mode (reflect constant, etc)

    Returns
    -------
    np.ndarray
        padded image (H' x W' x C)
    tuple
        number of tiles: (y_tiles, x_tiles)
    list[tuple]
        padding: [(0, y_pad), (0, x_pad)]
    """
    step = tile_size
    height, width = img.shape[:2]
    
    # Number of tiles
    x_tiles = int(np.floor(width/tile_size))
    y_tiles = int(np.floor(height/tile_size))

    # Residual for each dimension
    x_res = width - x_tiles * step     # Number of pixels not covered by
    y_res = height - y_tiles * step    # the tiles in each dimmension (residuals)
    
    if x_res == 0 and y_res == 0:
        return img, (y_tiles, x_tiles), [(0, 0), (0, 0)]
    
    if x_res > 0: x_tiles += 1
    if y_res > 0: y_tiles += 1
    
    x_pad = tile_size - x_res if x_res > 0 else 0
    y_pad = tile_size - y_res if y_res > 0 else 0
    
    img_padded = np.pad(img, [(0, y_pad), (0, x_pad), (0, 0)], mode=mode)
    
    return img_padded, (y_tiles, x_tiles), [(0, y_pad), (0, x_pad)]


def pad_lrtb(img, tile_size=256, step=128, mode='reflect'):
    """ Pad left/right and top/bottom.

    Pad an image on all edges: left, right, top, bottom:
        - (left/top) sides are padded by 'step', while
        - (right/bottom) sides are padded by the required amount

    Parameters
    ----------
    img : np.ndarray
        input image (H x W x C)
    tile_size : int
        tile size
    mode : string
        pading mode (reflect constant, etc)

    Returns
    -------
    np.ndarray
        padded image (H' x W' x C)
    tuple
        number of tiles: (y_tiles, x_tiles)
    list[tuple]
        padding used: [(y_pad0, y_pad1), (x_pad0, x_pad1)]
    """
    height, width = img.shape[:2]

    x_steps = int(np.ceil(width / step))     # Number of valid positions (with out considering
    y_steps = int(np.ceil(height / step))    # tile size; e.g. ceil(1000/256) = 4

    x_pad0 = step                                      # step * (x_steps - 1) -> last position for tile
    x_pad1 = step*(x_steps-1) + tile_size - width      # step*(x_steps-1) + size -> total size

    y_pad0 = step                                      # step*(x_steps-1) + size - width -> end padding
    y_pad1 = step*(y_steps-1) + tile_size - height

    width_ = x_pad0 + width + x_pad1      # Padded image width and height
    height_ = y_pad0 + height + y_pad1

    x_tiles = int(np.ceil((width_-tile_size+1)/step))
    y_tiles = int(np.ceil((height_-tile_size+1)/step))

    img_padded = np.pad(img, [(y_pad0, y_pad1), (x_pad0, x_pad1), (0, 0)], mode=mode)

    return img_padded, (y_tiles, x_tiles), [(y_pad0, y_pad1), (x_pad0, x_pad1)]


def unpad_img(img, pads, last=True):
    """ Unpad image

    Unpad the input image 'img' given 'pads' and return the unpadded image.

    Parameters
    ----------
    img : np.ndarray or torch.Tensor
        padded image
    pads : list[tuple]
        pads used for padding original image
    last : bool
        position of the "channels" dimension:
            last=True  -> H x W x C
            last=False -> C x H x W
        For inputs with no channel" dimension (H x W) any of the the two
        options yields a correct result.

    Returns
    -------
    np.ndarray or torch.Tensor
        unpadded image (with the original shape).
    """
    if last: h, w = img.shape[:2]
    else: h, w = img.shape[1:]

    y_pad0, y_pad1 = pads[0]
    x_pad0, x_pad1 = pads[1]
    
    height = h - (y_pad0 + y_pad1)
    width = w - (x_pad0 + x_pad1)
    
    if last: return img[y_pad0:y_pad0+height, x_pad0:x_pad0+width, ...]
    else: return img[..., y_pad0:y_pad0+height, x_pad0:x_pad0+width]
