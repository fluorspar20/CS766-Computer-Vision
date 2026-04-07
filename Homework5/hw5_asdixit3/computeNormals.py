import numpy as np


def computeNormals(light_dirs, img_list, mask):
    """
    Compute surface normals and albedo for an object using photometric stereo.

    Photometric stereo model (Lambertian reflectance):
        I_i = rho * (L_i . n)
    where I_i is the pixel intensity under light i, rho is the albedo,
    L_i is the i-th light vector (direction scaled by intensity), and n is
    the unit surface normal.

    This can be written as the linear system:
        I = S * g,   where g = rho * n
    Solve for g, then recover:
        rho = ||g||    (albedo)
        n   = g / rho  (unit normal)

    For pixels in the mask:
      - We have 5 light sources, but only the 3 that produce the highest
        brightness at each pixel should be used. (Using a shadowed or occluded
        light source produces erroneous normals.)
      - You may choose to use all 5 (over-determined system) for robustness,
        but using the top 3 is required.

    For background pixels (mask == False):
      - Set the normal to [0, 0, 1] (pointing toward the camera).
      - Set the albedo to 0.

    Scale the output albedo image so that all values lie in [0, 1].

    Parameters
    ----------
    light_dirs : numpy.ndarray, shape (5, 3)
        Light direction matrix from computeLightDirections. Each row is the
        [x, y, z] light vector (direction * intensity) for one light source.
    img_list : list of numpy.ndarray
        List of 5 object images (e.g., vase1..vase5) as uint8 arrays.
    mask : numpy.ndarray, shape (H, W), dtype bool
        Foreground mask from computeMask.

    Returns
    -------
    normals : numpy.ndarray, shape (H, W, 3)
        Surface normals. normals[r, c] = [nx, ny, nz] (unit vector).
        Background pixels have normal [0, 0, 1].
    albedo_img : numpy.ndarray, shape (H, W), dtype float64
        Albedo image scaled to [0, 1]. Background pixels are 0.
    """
    H, W = mask.shape
    normals = np.zeros((H, W, 3), dtype=np.float64)
    albedo_img = np.zeros((H, W), dtype=np.float64)

    img_stack = np.stack(img_list, axis=-1).astype(np.float64)
    for r in range(H):
        for c in range(W):
            if mask[r, c]:  # Foreground pixel
                I = img_stack[r, c, :]
                top3_indices = np.argsort(I)[-3:]
                S_top3 = light_dirs[top3_indices, :]
                I_top3 = I[top3_indices]

                g, _, _, _ = np.linalg.lstsq(S_top3, I_top3, rcond=None)
                rho = np.linalg.norm(g)
                if rho > 0:
                    n = g / rho
                else:
                    n = np.array([0, 0, 1])
                normals[r, c, :] = n
                albedo_img[r, c] = rho
            else:  # Background pixel
                normals[r, c, :] = [0, 0, 1]
                albedo_img[r, c] = 0.0
    
    # Scale albedo image to [0, 1]
    if np.max(albedo_img) > 0:
        albedo_img /= np.max(albedo_img)
    return normals, albedo_img
