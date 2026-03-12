import numpy as np
from tqdm import tqdm


def generate_hough_accumulator(img, theta_num_bins, rho_num_bins):
    """
    Generate a Hough accumulator array for an edge image.

    Parameters
    ----------
    img : ndarray (H, W)
        Edge image (nonzero pixels are treated as edges).
    theta_num_bins : int
        Number of bins for theta.
    rho_num_bins : int
        Number of bins for rho.

    Returns
    -------
    hough_img : ndarray (rho_num_bins, theta_num_bins)
        Hough accumulator normalized to 0-255.
    """

    # ---------------------------
    # START ADDING YOUR CODE HERE
    # ---------------------------

    H, W = img.shape

    # YOU CAN MODIFY/REMOVE THE PART BELOW IF YOU WANT
    # ------------------------------------------------
    # here we assume origin = middle of image, not top-left corner
    # you can fix the top-left corner too (just remove the part below)
    centre_x = np.floor(W/2)
    centre_y = np.floor(H/2)
    # x = x - centre_x;
    # y = y - centre_y;
    # ------------------------------------------------

    # Initialize accumulator
    hough_img = np.zeros((rho_num_bins, theta_num_bins), dtype=np.float64)

    # img is an edge image, find edge pixels
    row_idxs, col_idxs = np.nonzero(img)
    x = col_idxs - centre_x
    y = row_idxs - centre_y

    # Calculate rho and theta for the edge pixels
    diag_len = np.ceil(np.sqrt(centre_x**2 + centre_y**2))
    rhos = np.linspace(-diag_len, diag_len, rho_num_bins)
    thetas = np.linspace(0, np.pi, theta_num_bins)

    # Map to an index in the hough_img array
    # and accumulate votes.
    for i in tqdm(range(len(x))):
        xi = x[i]
        yi = y[i]

        for t_idx in range(theta_num_bins):
            theta = thetas[t_idx]
            rho = yi * np.cos(theta) - xi * np.sin(theta)

            # Find the corresponding rho index
            r_idx = np.argmin(np.abs(rhos - rho))

            hough_img[r_idx, t_idx] += 1

    # Normalize hough_img to 0-255
    hough_img = (hough_img / np.max(hough_img)) * 255
    hough_img = hough_img.astype(np.uint8)


    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    return hough_img
