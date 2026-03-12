import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import gray2rgb
from tqdm import tqdm


def line_finder(orig_img, hough_img, hough_threshold):
    """
    Detect lines from Hough accumulator and overlay them on the original image.

    Parameters
    ----------
    orig_img : ndarray (H, W) or (H, W, 3)
        Original grayscale or RGB image.
    hough_img : ndarray (rho_bins, theta_bins)
        Hough accumulator.
    hough_threshold : float
        Threshold above which Hough votes are considered strong.

    Returns
    -------
    line_detected_img : ndarray
        Annotated image with detected lines.
    """

    # Ensure image is RGB for drawing
    if orig_img.ndim == 2:
        img_rgb = gray2rgb(orig_img)
    else:
        img_rgb = orig_img.copy()

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    # --------------------------------------
    # TODO: START ADDING YOUR CODE HERE
    # --------------------------------------
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)

    (H, W) = orig_img.shape[:2]
    (N_rho, N_theta) = hough_img.shape

    centre_x = np.floor(W/2)
    centre_y = np.floor(H/2)
    diag_len = np.ceil(np.sqrt(centre_x**2 + centre_y**2))

    rhos = np.linspace(-diag_len, diag_len, N_rho)
    thetas = np.linspace(0, np.pi, N_theta)

    # Non-maximum suppression to find peaks
    neighborhood_size = 5
    half_size = neighborhood_size // 2
    candidate_rho_idxs, candidate_theta_idxs = np.where(hough_img > hough_threshold)
    
    strong_rho_idxs = []
    strong_theta_idxs = []

    for r_idx, t_idx in zip(candidate_rho_idxs, candidate_theta_idxs):
        r_start = max(0, r_idx - half_size)
        r_end = min(N_rho, r_idx + half_size + 1)
        t_start = max(0, t_idx - half_size)
        t_end = min(N_theta, t_idx + half_size + 1)

        if hough_img[r_idx, t_idx] == np.max(hough_img[r_start:r_end, t_start:t_end]):
            strong_rho_idxs.append(r_idx)
            strong_theta_idxs.append(t_idx)
    
    rho_idxs = np.array(strong_rho_idxs)
    theta_idxs = np.array(strong_theta_idxs)
    for r_idx, t_idx in zip(rho_idxs, theta_idxs):
        rho = rhos[r_idx]
        theta = thetas[t_idx]

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Generate two points on the line for plotting over the image
        if abs(cos_t) > abs(sin_t):
            # Line is more vertical, fix x and solve for y
            x1 = 0
            y1 = (rho + (x1 - centre_x) * sin_t) / cos_t + centre_y
            x2 = W
            y2 = (rho + (x2 - centre_x) * sin_t) / cos_t + centre_y
        else:
            # Line is more horizontal, fix y and solve for x
            y1 = 0
            x1 = ((y1 - centre_y) * cos_t - rho) / sin_t + centre_x
            y2 = H
            x2 = ((y2 - centre_y) * cos_t - rho) / sin_t + centre_x

        ax.plot((x1, x2), (y1, y2), 'r-', linewidth=2)

    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    # Convert figure to image array
    fig.canvas.draw()
    line_detected_img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    plt.close(fig)

    return line_detected_img
