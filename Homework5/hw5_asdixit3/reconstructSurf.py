"""
reconstructSurf.py — Surface reconstruction using the Frankot-Chellappa algorithm.

This file is provided as a demo. No submission is required for this function.
"""

import numpy as np


def reconstructSurf(normals, mask):
    """
    Reconstruct a depth map from surface normals using the Frankot-Chellappa
    algorithm (frequency-domain integration of surface gradients).

    Parameters
    ----------
    normals : numpy.ndarray, shape (H, W, 3)
        Surface normal map (output of computeNormals).
    mask : numpy.ndarray, shape (H, W), dtype bool
        Foreground mask (output of computeMask).

    Returns
    -------
    surf_img : numpy.ndarray, shape (H, W), dtype float64
        Reconstructed surface depth, normalized to [0, 1].
        Background pixels are set to 0.
    """
    eps = 1e-10

    # Compute surface gradients from normals: p = dz/dx, q = dz/dy
    p_img = normals[:, :, 0] / (normals[:, :, 2] + eps)
    q_img = normals[:, :, 1] / (normals[:, :, 2] + eps)

    # Fourier transform of gradients
    fp_img = np.fft.fft2(p_img)
    fq_img = np.fft.fft2(q_img)

    rows, cols = fp_img.shape

    # Build frequency grids (centered, then shift back)
    u = np.arange(cols) - cols // 2
    v = np.arange(rows) - rows // 2
    U, V = np.meshgrid(u, v)
    U = np.fft.ifftshift(U)
    V = np.fft.ifftshift(V)

    # Frankot-Chellappa integration in frequency domain
    fz = (1j * U * fp_img + 1j * V * fq_img) / (U ** 2 + V ** 2 + eps)

    # Inverse FFT back to spatial domain
    ifz = np.fft.ifft2(fz)
    ifz[~mask] = 0

    z = np.real(ifz)

    # Normalize to [0, 1]
    z_min, z_max = z.min(), z.max()
    surf_img = (z - z_min) / (z_max - z_min + eps)
    surf_img[~mask] = 0

    return surf_img
