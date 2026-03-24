import numpy as np


def compute_homography(src_pts_nx2, dest_pts_nx2):
    
    # TODO: Implement this function to compute the homography matrix H
    # that transforms src_pts_nx2 to dest_pts_nx2.

    n_points = src_pts_nx2.shape[0]
    A = np.zeros((2*n_points, 9))
    for i in range(n_points):
        x, y = src_pts_nx2[i, 0], src_pts_nx2[i, 1]
        xp, yp = dest_pts_nx2[i, 0], dest_pts_nx2[i, 1]
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        A[2*i + 1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        

    # Eigen decomposition of A^T A
    eigvals, eigvecs = np.linalg.eig(A.T @ A)
    idx = np.argmin(eigvals)
    h = eigvecs[:, idx]

    H_3x3 = h.reshape((3, 3))
    return H_3x3
