# -------------------------------------------------------------------------
# Walkthrough 1
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology


def hw2_walkthrough1():
    # -----------------
    # Convert a grayscale image to a binary image
    # -----------------
    fh1 = plt.figure()
    img = io.imread("coins.png")

    # Convert to grayscale if needed
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img.astype(np.float32) / 255.0

    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Original Image")

    # Convert the image into a binary image by applying a threshold
    threshold = 0.5
    bw_img = img_gray > threshold

    plt.subplot(1, 2, 2)
    plt.imshow(bw_img, cmap="gray")
    plt.title("Binary Image")

    plt.savefig("binary_coins.png")

    # -----------------
    # Remove noises in the binary image
    # -----------------
    # Clean the image (you may notice some holes in the coins) by using
    # dilation and then erosion
    fh2 = plt.figure()

    # Specify the number of dilations/erosions
    k = 10

    processed_img = bw_img.copy()
    for _ in range(k):
        processed_img = morphology.binary_dilation(processed_img, morphology.square(3))

    plt.subplot(1, 2, 1)
    plt.imshow(processed_img, cmap="gray")
    plt.title("After Dilation")

    for _ in range(k):
        processed_img = morphology.binary_erosion(processed_img, morphology.square(3))
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap="gray")
    plt.title("After Erosion")

    plt.savefig("noise_removal_coins.png")

    # -----------------
    # Remove the rices
    # -----------------
    # Apply erosion then dilation once to remove the rices
    fh3 = plt.figure()

    # Specify the number of erosions/dilations
    k = 15

    for _ in range(k):
        processed_img = morphology.binary_erosion(processed_img, morphology.square(3))

    plt.subplot(1, 2, 1)
    plt.imshow(processed_img, cmap="gray")
    plt.title("After Erosion")

    for _ in range(k):
        processed_img = morphology.binary_dilation(processed_img, morphology.square(3))

    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap="gray")
    plt.title("After Dilation")

    plt.savefig("morphological_operations_coins.png")

    plt.show()
    pass


if __name__ == "__main__":
    hw2_walkthrough1()
