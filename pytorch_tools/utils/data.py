import numpy as np
import cv2

from torch.utils.data import Dataset

import os
from tqdm import tqdm


def load_img_to_RAM(arr, base_dir='', to_RGB=False, flags=cv2.IMREAD_UNCHANGED, verbose=False):
    """ Load images into RAM

    Read the images indicated by 'arr' and store them in RAM. It is assumed
    that all images have:
    - the same shape, and
    - the same data type

    Parameters
    ----------
    arr : list
        array containing the path to each image
    base_dir : string
        base directory where images are contained. This is useful when the paths are
        relative and we want to specify the directory containing the images.
    to_RGB : bool
        wheter or not to convert the image to RGB
    flags : int
        flags used in cv2.imread

    Returns
    -------
    np.ndarray
        numpy array containing all the loaded images; shape:
        len(arr) x H x W x 3 
    """
    # Use first image to get the shape and type of the output array
    path = os.path.join(base_dir, arr[0])
    img = cv2.imread(path, flags)
    if img is None: raise ValueError(f'Error reading image: {path}')

    imgs = np.empty((len(arr), *img.shape), dtype=img.dtype)

    if verbose:
        print(f'Loading images (output array: {imgs.shape}) ...')
        pb = tqdm(total=len(arr)-1)

    # Fill output array
    imgs[0] = img
    for i in range(1, len(arr)):
        path = os.path.join(base_dir, arr[i])
        img = cv2.imread(path, flags)
        if img is None: raise ValueError(f'Error reading image: {path}')

        if to_RGB: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs[i] = img

        if verbose: pb.update()

    if verbose: pb.close()

    return imgs


class ImageMask_Dataset_RAM(Dataset):
    """ Dataset for (image, mask) paris

    Manage images and corresponding masks stored in RAM; i.e. both inputs are
    arrays (np.ndarray) """

    def __init__(self, imgs, msks, transform=None):
        """
        Parameters
        ----------
        imgs : np.ndarray
            images array (usually N x H x W x 3 - for segmentation)
        msks : np.ndarray
            masks array (usually N x H x W - for segmentation)
        transform : function
            transformation function applied to both images and masks. transform
            has the albumentations format: i.e., returns a dict with elements
            'image' and 'mask'.
        Dataset for tiled images. Training data is located
            in RAM. That is, 'imgs' and 'msks' are np.arrays of
            of size N x H x W x 3 and N x H x W. We assume that
            'transform' has the albumentations format; i.e. it
            return a dict """
        
        self.imgs = imgs
        self.msks = msks
        self.transform = transform
        
    def __len__(self):
        return len(self.msks)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        msk = self.msks[idx]
        
        if self.transform:
            transformed = self.transform(image=img, mask=msk)
            img = transformed['image']
            msk = transformed['mask']
            
        return img, msk
