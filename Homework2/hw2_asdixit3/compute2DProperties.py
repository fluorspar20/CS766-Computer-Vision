import numpy as np
from skimage import draw, color, util


def compute_2d_properties(orig_img, labeled_img):
    """
    Compute properties for each labeled object in the image.

    Parameters
    ----------
    orig_img : ndarray
        Original grayscale image.
    labeled_img : ndarray
        Labeled image (background = 0, labels 1..N).

    Returns
    -------
    db : ndarray
        Object database (rows = properties, cols = objects).
    out_img : ndarray
        RGB image with positions and orientations annotated.
    """

    # Collect properties for each object label > 0
    labels = np.unique(labeled_img)
    labels = labels[labels > 0]

    props = []
    for lab in labels:
        # rows, cols for current label
        r, c = np.nonzero(labeled_img == lab)
        area = len(r)
        r_bar = np.mean(r)
        c_bar = np.mean(c)

        # Central moments
        mu20 = np.sum((c - c_bar) ** 2)
        mu02 = np.sum((r - r_bar) ** 2)
        mu11 = np.sum((c - c_bar) * (r - r_bar))

        # Minimum and maximum moment of inertia
        common = np.sqrt((mu20 - mu02) ** 2 + 4 * (mu11 ** 2))
        Imin = 0.5 * ((mu20 + mu02) - common)
        Imax = 0.5 * ((mu20 + mu02) + common)

        # Orientation (degrees, clockwise from horizontal)
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        theta_deg = -np.degrees(theta)  # negative for clockwise

        # Roundness
        roundness = Imin / Imax if Imax != 0 else 0

        props.append([lab, r_bar, c_bar, Imin, theta_deg, roundness])
        pass

    # db should be a 2D numpy array with shape (6, num_objects)
    db = np.array(props).T if props else np.zeros((6, 0))

    # Annotate image: draw a dot at the center and a short line for orientation
    # Use skimage.draw.line or line_aa to draw on an RGB copy of orig_img
    out_img = color.gray2rgb(util.img_as_ubyte(orig_img))
    for prop in props:
      lab, r_bar, c_bar, Imin, theta_deg, roundness = prop
      rr, cc = draw.disk((r_bar, c_bar), radius=2, shape=out_img.shape[:2])
      out_img[rr, cc] = (255, 0, 0)
      line_len = 20
      dr = line_len * np.sin(np.radians(theta_deg))
      dc = line_len * np.cos(np.radians(theta_deg))
      rr, cc = draw.line(int(r_bar), int(c_bar), int(r_bar + dr), int(c_bar + dc))
      out_img[rr, cc] = (0, 255, 0)

    return db, out_img


if __name__ == "__main__":
    # Example:
    import imageio
    gray = imageio.imread("two_objects.png")
    labeled = imageio.imread("labeled_two_objects.png")
    db, annotated = compute_2d_properties(gray, labeled)
    imageio.imwrite("annotated_two_objects.png", annotated)
    pass
