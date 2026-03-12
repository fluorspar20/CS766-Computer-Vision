import numpy as np
from skimage import draw, color, util
from compute2DProperties import compute_2d_properties


def recognize_objects(orig_img, labeled_img, obj_db):
    """
    Recognize objects in labeled_img by comparing their properties with obj_db.

    Parameters
    ----------
    orig_img : ndarray
        Original grayscale image.
    labeled_img : ndarray
        Labeled image of the test scene.
    obj_db : ndarray
        Object database produced by compute_2d_properties.

    Returns
    -------
    output_img : ndarray
        RGB image showing the positions and orientations of recognized objects.
    """

    # extract properties from the labeled_img (similar to compute_2d_properties)
    # compare each object against obj_db using your criteria / thresholds

    test_db, _ = compute_2d_properties(orig_img, labeled_img)
    recognized = []
    
    # define thresholds for matching
    roundness_thresh = 0.05
    inertia_thresh = 0.20

    # iterate through each object found in the test image
    if test_db.shape[0] > 0:
        num_test_objs = test_db.shape[1]
        num_db_objs = obj_db.shape[1]

        for i in range(num_test_objs):
            curr_obj = test_db[:, i]
            
            # extract features for comparison - index 3 is Imin, index 5 is roundness
            curr_imin = curr_obj[3]
            curr_round = curr_obj[5]

            is_match = False
            
            # check against every object in the database
            for j in range(num_db_objs):
                db_imin = obj_db[3, j]
                db_round = obj_db[5, j]

                # compare roundness 
                diff_round = abs(curr_round - db_round)
                # compare moment of inertia
                diff_imin = abs(curr_imin - db_imin) / db_imin

                # If both properties are within tolerance, we found a match
                if diff_round < roundness_thresh and diff_imin < inertia_thresh:
                    is_match = True
                    break
            
            if is_match:
                recognized.append(curr_obj)
    
    # Visualize recognized objects on orig_img
    output_img = color.gray2rgb(util.img_as_ubyte(orig_img))
    # for each recognized object:
    #   draw dot and orientation line
    for obj in recognized:
        lab, r_bar, c_bar, Imin, theta_deg, roundness = obj
        rr, cc = draw.disk((r_bar, c_bar), radius=2, shape=output_img.shape[:2])
        output_img[rr, cc] = (255, 0, 0)
        line_len = 20
        dr = line_len * np.sin(np.radians(theta_deg))
        dc = line_len * np.cos(np.radians(theta_deg))
        rr, cc = draw.line(int(r_bar), int(c_bar), int(r_bar + dr), int(c_bar + dc))
        output_img[rr, cc] = (0, 255, 0)

    return output_img


if __name__ == "__main__":
    # Example usage:
    import imageio
    gray = imageio.imread("many_objects_1.png")
    labeled = imageio.imread("labeled_many_objects_1.png")
    db = np.load("object_db.npy")  # or load from mat/pkl
    output = recognize_objects(gray, labeled, db)
    imageio.imwrite("testing1c1_many_objects_1.png", output)
    pass
