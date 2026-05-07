import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import match_template


def compute_flow(img1, img2, search_radius, template_radius, grid_MN):
    # Check images have the same dimensions, and resize if necessary
    if img2.shape != img1.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Get number of rows and cols for output grid
    M = grid_MN[0]
    N = grid_MN[1]

    H, W = img1.shape[:2]
    # locations where we estimate the flow
    grid_y = np.round(np.linspace(template_radius + 1,
                      H - template_radius, M)).astype(int)
    grid_x = np.round(np.linspace(template_radius + 1,
                      W - template_radius, N)).astype(int)

    # allocate matrices where we will store the computed optical flow
    U = np.zeros((M, N))    # horizontal motion
    V = np.zeros((M, N))    # vertical motion

    # compute flow for each grid patch
    for i in range(M):
        for j in range(N):
            # ------------- PLEASE FILL IN THE NECESSARY CODE WITHIN THE FOR LOOP -----------------
            # Note: Wherever there are questions mark you should write
            # code and fill in the correct values there. You will need
            # to write more lines of code to obtain the correct values to
            # input in the questions marks.

            # extract the current patch/window (template)
            col = grid_x[j]
            row = grid_y[i]
            template = img1[row - template_radius : row + template_radius + 1, col - template_radius : col + template_radius + 1]
            # where we'll look for the template

            r_min = max(0, row - search_radius)
            r_max = min(H, row + search_radius + 1)
            c_min = max(0, col - search_radius)
            c_max = min(W, col + search_radius + 1)

            search_area = img2[r_min:r_max, c_min:c_max]

            # compute correlation
            if (search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]):
                continue
            corr_map = match_template(search_area, template)

            # Look at the correlation map and find the best match
            # The best match will have the Maximum Correlation value
            max_ind = np.argmax(corr_map)
            # Convert the index into row and col
            max_ind_row, max_ind_col = np.unravel_index(
                max_ind, corr_map.shape)

            # express peak location as offset from template location
            match_row = r_min + max_ind_row + template_radius
            match_col = c_min + max_ind_col + template_radius
            U[i, j] = match_col - col
            V[i, j] = match_row - row

    # Any post-processing or denoising needed on the flow

    # plot the flow vectors
    fig, ax = plt.subplots()
    ax.imshow(img1, cmap='gray')
    ax.quiver(grid_x, grid_y, U, -V, 2, color='y', linewidth=1.3)
    fig.canvas.draw()

    # Convert the figure directly into an image matrix
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    plt.close(fig)

    return img
