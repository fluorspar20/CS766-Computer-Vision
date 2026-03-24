import numpy as np
from computeHomography import compute_homography
from applyHomography import apply_homography


def run_ransac(Xs, Xd, ransac_n, eps):
    num_pts = Xs.shape[0]
    pts_id = np.arange(num_pts)
    inliers_id = np.array([])
    H = np.eye(3)  # H placeholder

    for iter in range(ransac_n):
        # ---------------------------
        # START ADDING YOUR CODE HERE
        # ---------------------------
        # 1. Randomly sample 4 points from pts_id and compute homography H using compute_homography.py
        # 2. Apply homography H to Xs and compute the distance between the transformed Xs and Xd
        # 3. Find inliers based on the distance and eps
        # 4. Keep track of the best homography and inliers based on the number of inliers

        ids = np.random.choice(pts_id, 4, replace=False)
        H = compute_homography(Xs[ids], Xd[ids])
        Xs_transformed = apply_homography(H, Xs)
        distances = np.linalg.norm(Xs_transformed - Xd, axis=1)
        current_inliers_id = pts_id[distances < eps]
        if len(current_inliers_id) > len(inliers_id):
            inliers_id = current_inliers_id

        # ---------------------------
        # END ADDING YOUR CODE HERE
        # ---------------------------
        pass  # placeholder so for loop isn't empty.

    return inliers_id, H
