import numpy as np
from scipy.signal import convolve2d
from skimage import img_as_float


def generate_index_map(gray_stack, w_size):
    H, W, N = gray_stack.shape

    # Compute the focus measure -- the sum-modified laplacian
    #
    # horizontal Laplacian kernel
    Kx = np.array([[0.25, 0, 0.25],
                   [1,   -3,  1],
                   [0.25, 0, 0.25]])
    Ky = Kx.T   # vertical version

    # horizontal and vertical Laplacian responses
    Lx = np.zeros((H, W, N))
    Ly = np.zeros((H, W, N))
    for n in range(N):
        I = img_as_float(gray_stack[:, :, n])
        Lx[:, :, n] = convolve2d(I, Kx, mode="same", boundary="symm")
        Ly[:, :, n] = convolve2d(I, Ky, mode="same", boundary="symm")

    # sum-modified Laplacian
    SML = (np.abs(Lx) ** 2) + (np.abs(Ly) ** 2)
    # can also use the absolute value itself
    # this is probably more well-known
    # SML = np.abs(Lx) + np.abs(Ly)

    ###########################################################################
    K = np.ones((2 * w_size + 1, 2 * w_size + 1))
    for n in range(N):
        SML[:, :, n] = convolve2d(SML[:, :, n], K, mode="same", boundary="symm")
    ###########################################################################

    ###########################################################################
    index_map = np.argmax(SML, axis=2)
    return index_map
