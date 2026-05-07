import os
import glob
import numpy as np
from skimage import io, color


def load_focal_stack(focal_stack_dir):

    ###########################################################################
    
    filenames = sorted(glob.glob(os.path.join(focal_stack_dir, "*.jpg")))

    rgb_list = []
    gray_list = []
    for filename in filenames:
        img = io.imread(filename)
        rgb_list.append(img)
        gray_list.append(color.rgb2gray(img))

    rgb_stack = np.concatenate(rgb_list, axis=-1)
    gray_stack = np.stack(gray_list, axis=-1)

    ###########################################################################

    return rgb_stack, gray_stack
