import numpy as np
import pydicom as dcm

import os

def read_DICOM_dir(path, rescale=True, reverse=False, verbose=False):
    """ Read all DICOM files contained in a directory

    Parameters
    ----------
    path : string
        directory containing DICOM files

    rescale : bool
        whether or not to rescale the DICOM image using the slope and
        intercept; i.e., img = slope*img + intercept

    verbose : bool
        verbosity

    Returns
    -------
    np.ndarray
        3D slice; D x H x W (D = depth)
    tuple
        slice thickness, pixel spacing y, pixel spacing x
    tuple
        window parameters: (center, width)
    """
    files = os.listdir(path)    # Get all files (arbitrary order)
    if verbose: print(f'Reading DICOM files...')
    dcm_files = [dcm.dcmread(os.path.join(path, f)) for f in files]

    # Sort DICOM files according to their 'Instance Number'
    dcm_files = sorted(dcm_files, key=lambda x: int(x.InstanceNumber), reverse=reverse)

    # Get properties from first DICOM file
    slope, intercept = dcm_files[0].RescaleSlope, dcm_files[0].RescaleIntercept
    center, width = dcm_files[0].WindowCenter, dcm_files[0].WindowWidth
    dz = float(dcm_files[0].SliceThickness)
    dy, dx = [float(a) for a  in dcm_files[0].PixelSpacing]

    # Create 3D array of slices
    slices = np.empty((len(files), *dcm_files[0].pixel_array.shape),
                      dtype=dcm_files[0].pixel_array.dtype)

    if verbose:
        print(f'Tickness, PixelSpacing: {dz}, ({dy}, {dx})')
        print(f'Slope, intercept: {slope}, {intercept}')
        print(f'Window center, width: {center}, {width}')
        print(f'3D slice: {slices.shape}')

    for i in range(len(dcm_files)):
        slices[i] = dcm_files[i].pixel_array

    if rescale: slices = slope*slices + intercept
    return slices, (dz, dy, dx), (center, width)


def read_DICOM(path, rescale=True):
    """ Read a single DICOM file

    Parameters
    ----------
    path : string
        path to DICOM file; e.g. 'path/to/133.dcm'

    rescale : bool
        whether or not to rescale the DICOM image using the slope and
        intercept; i.e., img = slope*img + intercept

    Returns
    -------
    np.ndarray
        2D slice; H x W (D = depth)
    """
    dcm_file = dcm.dcmread(path)

    # Get properties from first DICOM file
    slope, intercept = dcm_file.RescaleSlope, dcm_file.RescaleIntercept
    img = dcm_file.pixel_array

    if rescale: img = slope*img + intercept

    return img


def window(img, center, width):
    """ Window input image:

    Output image:
        
        image = clip(image, center-width/2, center+width/2)

    Parameters
    ----------
    img : np.ndarray
        input image

    center : int or float
        window center

    width : int or float
        window width

    Returns
    -------
    np.ndarray
        windowed image
    """
    return np.clip(img, center-width/2, center+width/2)

def window_normalize(img, center, width, float32=True):
    """ Window and normalize input image

    Output image:

        low = center - width/2
        high = center + width/2

        image = clip(image, low, high)
        image = (image - low) / (high - low)


    Parameters
    ----------
    img : np.ndarray
        input image, H x W

    center : int or float
        window center

    width : int of float
        window width

    float32 : bool
        returned image is np.float32

    Returns
    -------
    np.ndarray
        windowed and normalized image
    """
    a = center - width/2
    b = center + width/2
    clip = np.clip(img, a, b)
    clip = (clip - a) / (b - a)
    
    if float32 and clip.dtype != np.float32:
        return clip.astype(np.float32)
    else:
        return clip
