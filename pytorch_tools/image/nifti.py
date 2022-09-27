import nibabel as nib

def read_NIFTI(path, transform=None):
    """ Read NIFTI file

    Parameters
    ----------
    path : string
        path to .nii file
    transform : function
        if not None, function used to transform the read NIFTI image.
    """
    nifti = nib.load(path)
    data = nifti.get_fdata()

    if transform: data = transform(data)

    return data
