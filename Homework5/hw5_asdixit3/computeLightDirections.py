import numpy as np


def computeLightDirections(center, radius, img_list):
    """
    Compute the direction and intensity of each of the 5 light sources using
    images of a known sphere.

    For a Lambertian sphere under a single distant light source, the brightest
    point on the sphere corresponds to the surface normal that best aligns with
    the light direction. Use this property to recover each light source direction.

    Steps for each image:
      1. Find the brightest pixel (x, y) in the image.
      2. Compute the surface normal at that pixel using the sphere center and
         radius (see the derivation in your README).
      3. Scale the unit normal vector so that its magnitude equals the brightness
         of the brightest pixel (this encodes the light source intensity).

    Normal vector formula (to include in your README):
        Given pixel (x, y) and sphere center (cx, cy) with radius r:
            nx = (x - cx) / r
            ny = (y - cy) / r
            nz = sqrt(1 - nx^2 - ny^2)
        where the coordinate system has x- and y-axes parallel to the image
        axes, and the z-axis pointing toward the camera (left-hand system).

    Why is it safe to assume the brightest point gives the light direction?
    (Answer in your README.)

    Parameters
    ----------
    center : numpy.ndarray, shape (2,)
        [cx, cy] — sphere center in image coordinates (from findSphere).
    radius : float
        Sphere radius in pixels (from findSphere).
    img_list : list of numpy.ndarray
        List of 5 images (sphere1..sphere5) as uint8 arrays.

    Returns
    -------
    light_dirs_5x3 : numpy.ndarray, shape (5, 3)
        Row i contains the [x, y, z] components of the light vector for source i.
        The magnitude of each row encodes the light source intensity.
    """
    light_dirs = []
    for img in img_list:
        # Find the brightest pixel
        y, x = np.unravel_index(np.argmax(img), img.shape)
        brightness = img[y, x]

        nx = (x - center[0]) / radius
        ny = (y - center[1]) / radius
        nz = np.sqrt(max(0, 1 - nx**2 - ny**2))
        normal = np.array([nx, ny, nz])

        light_dir = normal * brightness
        light_dirs.append(light_dir)
    return np.array(light_dirs)