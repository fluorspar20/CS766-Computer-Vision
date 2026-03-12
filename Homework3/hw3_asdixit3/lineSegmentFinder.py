import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from skimage import feature

def line_segment_finder(orig_img, hough_img, hough_threshold):
    """
    Detect line segments from Hough accumulator and draw them on the original image.

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
    cropped_line_img : ndarray
        Annotated image with detected line segments.
    """
    # Ensure image is RGB
    if orig_img.ndim == 2:
        img_rgb = gray2rgb(orig_img)
    else:
        img_rgb = orig_img.copy()

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    # --------------------------------------
    # START ADDING YOUR CODE HERE
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

    gray_img = orig_img if orig_img.ndim == 2 else orig_img[:,:,0]
    edge_img = feature.canny(gray_img, sigma=2.0, low_threshold=0.04*np.max(gray_img), high_threshold=0.08*np.max(gray_img))  
    y_idxs, x_idxs = np.nonzero(edge_img)
    x_c = x_idxs - centre_x
    y_c = y_idxs - centre_y

    # Non-Maximal Suppression
    neighborhood_size = 5
    half_size = neighborhood_size // 2
    
    cand_rhos, cand_thetas = np.where(hough_img > hough_threshold)
    
    strong_rho_idxs = []
    strong_theta_idxs = []

    for r, c in zip(cand_rhos, cand_thetas):
        r_min = max(0, r - half_size)
        r_max = min(N_rho, r + half_size + 1)
        c_min = max(0, c - half_size)
        c_max = min(N_theta, c + half_size + 1)
        
        neighborhood = hough_img[r_min:r_max, c_min:c_max]
        
        if hough_img[r, c] == np.max(neighborhood):
            strong_rho_idxs.append(r)
            strong_theta_idxs.append(c)

    for r_idx, t_idx in zip(strong_rho_idxs, strong_theta_idxs):
        rho = rhos[r_idx]
        theta = thetas[t_idx]

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        distances = np.abs(x_c * sin_t - y_c * cos_t + rho)
        inliers_mask = distances < 2 
        
        if np.sum(inliers_mask) > 0:
            inlier_x = x_idxs[inliers_mask]
            inlier_y = y_idxs[inliers_mask]
            inlier_x_c = x_c[inliers_mask]
            inlier_y_c = y_c[inliers_mask]
                
            # project inliers onto the line to find their position on the line
            projections = inlier_x_c * cos_t + inlier_y_c * sin_t
            
            # sort the points along the line
            sort_idxs = np.argsort(projections)
            sorted_projs = projections[sort_idxs]
            sorted_x = inlier_x[sort_idxs]
            sorted_y = inlier_y[sort_idxs]
            
            # find gaps > 10 pixels and split the line
            gaps = np.diff(sorted_projs)
            max_gap_threshold = 10
            split_indices = np.where(gaps > max_gap_threshold)[0] + 1
            
            segments_x = np.split(sorted_x, split_indices)
            segments_y = np.split(sorted_y, split_indices)
            
            for seg_x, seg_y in zip(segments_x, segments_y):
                if len(seg_x) > 35:  # if the segment is long enough to be a real edge
                    pt1_x, pt1_y = seg_x[0], seg_y[0]
                    pt2_x, pt2_y = seg_x[-1], seg_y[-1]
                    ax.plot([pt1_x, pt2_x], [pt1_y, pt2_y], color='lime', linewidth=2.0)

    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    fig.canvas.draw()
    line_detected_img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    plt.close(fig)

    return line_detected_img