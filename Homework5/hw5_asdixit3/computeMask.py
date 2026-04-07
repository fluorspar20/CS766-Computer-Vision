import numpy as np


def computeMask(img_list):
    """
    Compute a binary foreground mask for an object.

    A pixel belongs to the foreground (mask = True) if it has a nonzero value
    in at least one of the 5 input images. A pixel is background (mask = False)
    only if it is zero in ALL 5 images.

    Parameters
    ----------
    img_list : list of numpy.ndarray
        List of 5 images of the object (e.g., vase1..vase5) as uint8 arrays.

    Returns
    -------
    mask : numpy.ndarray, shape (H, W), dtype bool
        Binary foreground mask. True = object, False = background.
    """
    stacked_imgs = np.stack(img_list, axis=-1)
    mask = np.any(stacked_imgs > 0, axis=-1)
    return mask
