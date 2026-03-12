# runHw2.py
# runHw2 is the "main" interface that lets you execute all the
# walkthroughs and challenges in homework 2. It lists a set of
# functions corresponding to the problems that need to be solved.
#
# Note that this file also serves as the specifications for the
# functions you are asked to implement. In some cases, your submissions
# will be autograded. Thus, it is critical that you adhere to all the
# specified function signatures.
#
# Before your submission, make sure you can run
#   python runHw2.py all
# without any error.
#
# Usage:
# python runHw2.py                     : list all the registered functions
# python runHw2.py 'method name'       : execute a specific test
# python runHw2.py all                 : execute all the registered functions

from ctypes import util
import sys
from pathlib import Path
import imageio
import numpy as np
from skimage import color, util

from signAcademicHonestyPolicy import sign_academic_honesty_policy
from hw2_walkthrough1 import hw2_walkthrough1
from generateLabeledImage import generate_labeled_image
from compute2DProperties import compute_2d_properties
from recognizeObjects import recognize_objects


def runHw2(*args):
    fun_handles = {
        "honesty": honesty,
        "walkthrough1": walkthrough1,
        "challenge1a": challenge1a,
        "challenge1b": challenge1b,
        "challenge1c1": challenge1c1,
        "challenge1c2": challenge1c2,
    }
    runTests(args, fun_handles)


def honesty():
    """State your agreement to the academic honesty policy."""

    # Type your full name and netID (both in string) to state your agreement
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy("Amogh Sudhir Dixit", "asdixit3")


def walkthrough1():
    # Complete hw2_walkthrough1.py.
    hw2_walkthrough1()


def challenge1a():
    # Convert gray-level images to labeled images using a single threshold.
    img_list = ["two_objects", "many_objects_1", "many_objects_2"]
    threshold_list = [127, 127, 127]  # use the same value for all images
    for name, thr in zip(img_list, threshold_list):
        gray_img = imageio.imread(f"{name}.png")
        labeled_img = generate_labeled_image(gray_img, thr)
        # Save labeled image ("labeled_<image_name>.png")
        imageio.imwrite(f"labeled_{name}.png", labeled_img.astype(np.uint8))
        # Save a colored visualization of labeled image (label2rgb equivalent, "rgb_labeled_<image_name>.png")
        num_labels = labeled_img.max() + 1
        colors = np.random.rand(num_labels, 3)
        colors[0] = [0, 0, 0] # Background black
        # Map labels to colors
        rgb_labeled = colors[labeled_img]
        imageio.imwrite(f"rgb_labeled_{name}.png", util.img_as_ubyte(rgb_labeled))
    pass


def challenge1b(): 
    # Use compute_2d_properties to build an object database from two_objects.png
    labeled_two_obj = imageio.imread("labeled_two_objects.png")
    orig_img = imageio.imread("two_objects.png")
    obj_db, out_img = compute_2d_properties(orig_img, labeled_two_obj)
    # Save the annotated image and the database for later use ("annotated_two_objects.png", "object_db.npy", "object_db.npy")
    imageio.imwrite("annotated_two_objects.png", out_img)
    np.save("object_db.npy", obj_db)
    pass


def challenge1c1():
    # Load the database created in challenge1b and recognize objects
    img_list = ["many_objects_1", "many_objects_2"]
    obj_db = np.load("object_db.npy")
    for name in img_list:
        labeled_img = imageio.imread(f"labeled_{name}.png")
        orig_img = imageio.imread(f"{name}.png")
        output_img = recognize_objects(orig_img, labeled_img, obj_db)
        # Save the output image ("testing1c1_<image_name>.png")
        imageio.imwrite(f"testing1c1_{name}.png", output_img)

    pass


def challenge1c2():
    # Build the database from many_objects_1.png
    db_img = "many_objects_1"
    labeled_img = imageio.imread(f"labeled_{db_img}.png")
    orig_img = imageio.imread(f"{db_img}.png")
    obj_db, out_img = compute_2d_properties(orig_img, labeled_img)
    # Save the annotated image and the database for later use ("annotated_many_objects_1.png", "object_db_many_objects_1.npy")
    imageio.imwrite(f"annotated_{db_img}.png", out_img)
    np.save(f"object_db_{db_img}.npy", obj_db)

    img_list = ["two_objects", "many_objects_2"]
    for name in img_list:
        labeled_img = imageio.imread(f"labeled_{name}.png")
        orig_img = imageio.imread(f"{name}.png")
        output_img = recognize_objects(orig_img, labeled_img, obj_db)
        imageio.imwrite(f"testing1c2_{name}.png", output_img)

    pass



def runTests(args, fun_handles):
    if not args:
        print("Registered functions:")
        for f in fun_handles:
            print(" -", f)
        return

    arg = args[0]
    if arg == "all":
        for name, func in fun_handles.items():
            if name.startswith("demo"):
                continue
            print(f"Running {name}()...")
            func()
    elif arg in fun_handles:
        print(f"Running {arg}()...")
        fun_handles[arg]()
    else:
        print("Unknown function name:", arg)


# Allow running from command line
if __name__ == "__main__":
    runHw2(*sys.argv[1:])
