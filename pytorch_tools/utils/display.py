import numpy as np
import matplotlib.pyplot as plt

import cv2

def show_img_with_tiles(img, tiles, figsize, cmap='gray'):
    """ Show image next a tiled image.

    i.e., tiles     image
         -----     -----
         i i i |
         i i i |   I
         i i i |

    Parameters
    ----------
    img : np.ndarray
        image (H x W x C)
    tiles : np.ndarray
        tiled image (V x U x H x W x C)
    figsize : tuple
        figure size (width, height)
    cmap : string
        colormap
    """
    y_tiles, x_tiles = tiles.shape[:2]

    fig = plt.figure(constrained_layout=False, figsize=figsize)
    subfigs = fig.subfigures(1, 2, width_ratios=[1,1])

    # Left Image: Display Tiles
    ax_left = subfigs[0].subplots(y_tiles, x_tiles)
    for j in range(y_tiles):
        for i in range(x_tiles):
            ax_left[j,i].imshow(tiles[j,i], cmap=cmap)
            ax_left[j,i].set_axis_off()

    # Right Image: Display Image
    ax_right = subfigs[1].subplots(1, 1)
    ax_right.imshow(img, cmap=cmap)
    ax_right.set_axis_off()

    plt.show()

# Example of how to easily construct a cat2color: label -> rgb object:
#
# from matplotlib import cm
# viridis = cm.get_cmap('viridis', 4)
#
# Using matplotlib.cm.get_cmap, we can get a colormap object.
# virids is ListedColormap object that divides the viridis color space
# into 4 pieces (colorspace ranges from [0,1])
# e.g
#       0        0.25      0.5       0.75      1.0
#       | color 1 | color 2 | color 3 | color 4 |
#
# To access a color we simple call:
#   viridis(0.1) -> returns 'color 1' since 0.1 in [0, 0.25)
#   returned is color is in RGBa format

# example:
# viridis = cm.get_cmap('viridis', nLabels)
# cat2color = {i: viridis(i/nLabels)[:3] for i in range(nLabels)}

def paint_mask(mask, labels, cat2color, background=0):
    """ Paint mask

    Given a mask (labeled or one-hot), return an image where each label is
    mapped to a different color; 'cat2color' defines a map (int -> rgb)

        label   -> color, or
        channel -> color (for one-hot encoded masks)

    Parameters
    ----------
    mask : np.ndarray
        single-channel integer image (H, W), or
        multiple-channel one-hot encoded image (H, W, N)
    labels : list
        list of labels to display, for labeled mask, or
        list of channels to display, for ont-hot mask
    cat2color : dict(int, np.ndarray)
        map label(chennel) -> color (int -> rgb);
        int -> np.ndarray
    background : np.ndarray
        background value

    Returns
    -------
    np.ndarray
        colored image (H, W, 3) same dtype as cat2color dtype
    """
    dtype = cat2color[list(cat2color.keys())[0]].dtype
    canvas = np.empty((*mask.shape[:2], 3), dtype=dtype)
    canvas[:] = background

    if len(mask.shape) == 2:
        for i in labels:
            canvas[mask == i] = cat2color[i]
    else:
        for i in labels:
            canvas[mask[...,i] != 0] = cat2color[i]

    return canvas

def overlay_paintedMask(img, mask, labels, cat2color):
    """ Overlay the painted mask on top of the provided image

    It is assumed that background color = (0, 0, 0).

    Note: img.dtype should be the same as cat2color[keys].dtype.

    Parameters
    ----------
    img : np.ndarray
        input image
    mask : np.ndarray
        single-channel integer image (H, W), or
        multiple-channel one-hot encoded image (H, W, N)
        single-channel labeled image; shape (H, W)
    labels : list
        list of labels to display, for labeled mask, or
        list of channels to display, for ont-hot mask
    cat2color : dict(int, np.ndarray)
        map int -> np.ndarray.
    """
    ret = paint_mask(mask, labels, cat2color, background=0)    # Paint mask

    canvas = img.copy()
    if len(canvas.shape) == 2:    # If gray scale, convert to RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # Set all places where there is a label to the corresponding color
    canvas[ret != 0] = ret[ret != 0]
    return canvas



def hist_cv_show_short(hist):
    """ Display Cross-Validation history (short)

    Compute the average value up the shortest array in 'hist'. That is, if
    len(average) = min(hist[i]) for all i. For example,

        [1, 2, 3, 4] and [2, 3] -> [3/2, 5/2] = [1.5, 2.5]

    Parameters
    ----------
    hist : list[list]
        array of arrays containing the history of the cross-validation
        metrics. Each array in hist can be of different length.
    """
    for k in range(len(hist)):
        x = np.arange(1, len(hist[k])+1)
        plt.plot(x, hist[k], alpha=0.2, label=f'fold-{k+1}')
        
    # Array with normalized lengts
    min_len = min([len(arr) for arr in hist])
    arr = np.array([arr[:min_len] for arr in hist])
    arr_mean = arr.mean(axis=0)
    arr_std = arr.std(axis=0)
    
    x = np.arange(1, min_len+1)
    plt.plot(x, arr.mean(axis=0), alpha=1.0, lw=2, label='mean')
    
    plt.fill_between(
        x,
        arr_mean + arr_std,
        arr_mean - arr_std,
        color='grey',
        alpha=0.2,
        label=r'$\pm$ 1 std. dev.',
    )
    
    plt.legend()
    plt.show()

def hist_cv_show_long(hist):
    """ Display Cross-Validation history (long)

    Compute the average value up to the longest array in 'hist'. That is, if
    len(hist[i]) < len(hist[j]) do not take into consideration hist[i] when
    computing the average. For example

        [1, 2, 3, 4] and [2, 3] -> [3/2, 5/2, 3/1, 4/1] = [1.5, 2.5, 3, 4]

    Parameters
    ----------
    hist : list[list]
        array of arrays containing the history of the cross-validation
        metrics. Each array in hist can be of different length.
    """
        
    max_len = max([len(arr) for arr in hist])
    arr = np.NaN * np.ones((len(hist), max_len))
    for i in range(len(hist)):
        arr[i,:len(hist[i])] = hist[i]
        
    x = np.arange(1, max_len+1)    
    for k in range(len(arr)):
        plt.plot(x, arr[k], alpha=0.2, label=f'fold-{k+1}')
    
    arr_mean = np.nanmean(arr, axis=0)
    arr_std = np.nanstd(arr, axis=0)
    plt.plot(x, arr_mean, alpha=1.0, lw=2, label='mean')
    
    plt.fill_between(
        x,
        arr_mean + arr_std,
        arr_mean - arr_std,
        color='grey',
        alpha=0.2,
        label=r'$\pm$ 1 std. dev.',
    )
    
    plt.legend()
    plt.show()
