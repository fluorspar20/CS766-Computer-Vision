import numpy as np
from skimage import color


def generate_labeled_image(gray_img, threshold):
    """
    Convert a gray-level image to a binary image using a threshold value and
    segment the binary image into several connected regions.

    Parameters
    ----------
    gray_img : ndarray
        Input gray-level image (2D) or RGB image (will be converted to grayscale).
    threshold : float or int
        Threshold value. Use a single value for all provided images.

    Returns
    -------
    labeled_img : ndarray
        Labeled image with background = 0 and labels 1..N for each object.
    """

    # Ensure grayscale
    if gray_img.ndim == 3:
        gray = color.rgb2gray(gray_img)
    else:
        gray = gray_img

    # Convert to binary using the threshold
    binary = gray > threshold

    # Label connected components (8-connectivity) - implement your own version of connected component labeling using region growing

    labeled_img = np.zeros_like(binary, dtype=np.int32)
    label = 1
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i, j] and labeled_img[i, j] == 0:
                # Start a new region
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if (0 <= x < binary.shape[0] and 0 <= y < binary.shape[1] and
                            binary[x, y] and labeled_img[x, y] == 0):
                        labeled_img[x, y] = label
                        # Add neighbors (8-connectivity)
                        stack.extend([(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]])

                label += 1

                
    # Make sure labels are consecutive integers starting at 1, background 0
    labeled_img = labeled_img.astype(np.uint8)

    return labeled_img


if __name__ == "__main__":
    # Example usage (fill in the missing pieces)
    import imageio
    img = imageio.imread("many_objects_2.png")
    labeled = generate_labeled_image(img, 127)
    print(labeled)
    imageio.imwrite("labeled_many_objects_2.png", labeled.astype(np.uint8))
    pass
