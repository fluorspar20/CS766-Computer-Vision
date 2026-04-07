import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops



def findSphere(img):
    """
    Locate the sphere in an image and compute its center and radius.

    Under orthographic projection, the sphere projects to a circle on the image
    plane. Find the circle by thresholding the image, computing the centroid of
    the foreground region, and estimating the radius from the area.

    You may use scikit-image functions such as:
        skimage.filters.threshold_otsu
        skimage.measure.label
        skimage.measure.regionprops

    Parameters
    ----------
    img : numpy.ndarray
        Grayscale or RGB image of the sphere (float64, values in [0, 1]).
        Use the image sphere0.png, which is illuminated from multiple directions
        so the entire front hemisphere is visible.

    Returns
    -------
    center : numpy.ndarray, shape (2,)
        [cx, cy] — the x (column) and y (row) coordinates of the sphere center
        in image coordinates.
    radius : float
        Radius of the sphere's circular projection in pixels.
    """    
    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img

    thresh = threshold_otsu(img_gray)
    binary_mask = img_gray > thresh
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    
    largest_region = max(regions, key=lambda r: r.area)
    center = np.array(largest_region.centroid)[::-1]  # (row, col) to (x, y)
    radius = np.sqrt(largest_region.area / np.pi)
    return center, radius