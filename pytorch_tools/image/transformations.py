import numpy as np

def resize_keepAspectRatio(img, target=512, value=0, center=True, interpolation=cv2.INTER_LINEAR):
    """ Resize image to (target, target) while keeping aspect ratio

    For example, if input image is (768, 512) and target=512, then resized
    image will be of size (512, 341), remaining pixels will then be filled with
    value to generate a (512, 512) image.

    Parameters
    ----------
    img : np.ndarray
        input image; H x W x C or (H x W)

    target : int
        target size, always squared: height=target, width=target.

    value : int or float
        background value

    center : bool
        whether or not to center the resized image (only useful when one of
        the sides of the resized image is smaller than target size).

    interpolation : int (cv2.INTER_xxx)
        interpolation used in image resize

    Returns
    -------
    np.ndarray
        resized image
    """
    height, width = img.shape[:2]    # img: H x W x C (or H x W)

    r = min(target/height, target/width)
    h = round(r*height)
    w = round(r*width)

    tmp = cv2.resize(img, (w,h), interpolation=interpolation)

    ret = value * np.ones((target, target), dtype=img.dtype)

    if center:
        pos_y = int(max(target - h, 0)/2)
        pos_x = int(max(target - w, 0)/2)
        ret[pos_y:pos_y+h, pos_x:pos_x+w] = tmp
    else:
        ret[:h, :w] = tmp

    return ret
