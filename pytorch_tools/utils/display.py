import numpy as np

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

def paint_mask(mask, labels, cat2color, dark=True):
    """ Paint mask

    Given a labeled image, return an image where each label is mapped to a
    different color. 'cat2color' define a map: label -> color (int -> rgb)

    Parameters
    ----------
    mask : np.ndarray
        single-channel integer image, shape (H, W)
    n_cat : int
        number of categories
    cat2color : dict
        map label -> color (int -> rgb)
    dark : bool
        background, either: dark (True) | light (False)

    Returns
    -------
    np.ndarray
        colored image (H, W, 3) - np.float32 """
    if dark:
        canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    else:
        canvas = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    for i in labels:
        canvas[mask == i] = cat2color[i]

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
