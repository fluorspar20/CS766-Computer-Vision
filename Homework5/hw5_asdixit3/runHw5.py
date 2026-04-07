"""
runHw5.py - Main interface for Homework 5: Photometric Stereo

Usage:
    python runHw5.py               # list all registered functions
    python runHw5.py all           # execute all challenges
    python runHw5.py challenge1a   # execute a specific challenge

Before submission, make sure you can run all challenges with:
    python runHw5.py all
with no errors.
"""

import sys
import os
import numpy as np
from PIL import Image

# Directory where images are located
IMG_DIR = ''

def load_img(filename):
    """Load an image as a float64 numpy array in [0, 1]."""
    path = os.path.join(IMG_DIR, filename)
    return np.array(Image.open(path)).astype(np.float64) / 255.0

def load_img_uint8(filename):
    """Load an image as a uint8 numpy array."""
    path = os.path.join(IMG_DIR, filename)
    return np.array(Image.open(path))

def save_img(img, filename):
    """Save a float [0,1] or uint8 image to the current directory."""
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if img.dtype != np.uint8:
        img_save = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_save = img
    Image.fromarray(img_save).save(out_path)
    print(f"  Saved: {filename}")

# -------------------------------------------------------------------------
# Academic Honesty Policy
# -------------------------------------------------------------------------
def honesty():
    """Sign the academic honesty policy."""
    name = "Amogh Sudhir Dixit"
    net_id = "asdixit3"
    # TODO: Replace with your actual name and net ID
    if name == "Your Full Name" or net_id == "yournetid":
        raise ValueError("Please fill in your name and net ID in the honesty() function.")
    print("***********************")
    print(f"I, {name} ({net_id}),")
    print("certify that I have read and agree to the Code of Academic Integrity.")
    print("***********************")

# -------------------------------------------------------------------------
# Challenge 1a: Find the sphere
# -------------------------------------------------------------------------
def challenge1a():
    """Compute the properties of the sphere (center and radius)."""
    from findSphere import findSphere

    img = load_img('sphere0.png')
    center, radius = findSphere(img)
    print(f"  Center: {center}")
    print(f"  Radius: {radius:.2f}")

    # Save results for later use
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sphere_properties.npy'),
            {'center': center, 'radius': radius}, allow_pickle=True)

# -------------------------------------------------------------------------
# Challenge 1b: Compute light source directions
# -------------------------------------------------------------------------
def challenge1b():
    """Compute the directions and intensities of the 5 light sources."""
    from computeLightDirections import computeLightDirections

    data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sphere_properties.npy'),
                   allow_pickle=True).item()
    center = data['center']
    radius = data['radius']

    img_list = [load_img_uint8(f'sphere{i}.png') for i in range(1, 6)]
    light_dirs_5x3 = computeLightDirections(center, radius, img_list)
    print("  Light directions (5x3):")
    print(light_dirs_5x3)

    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'light_dirs.npy'),
            light_dirs_5x3)

# -------------------------------------------------------------------------
# Challenge 1c: Compute object mask
# -------------------------------------------------------------------------
def challenge1c():
    """Compute the binary foreground mask for the vase."""
    from computeMask import computeMask

    vase_img_list = [load_img_uint8(f'vase{i}.png') for i in range(1, 6)]
    mask = computeMask(vase_img_list)
    print(f"  Mask shape: {mask.shape}, foreground pixels: {np.sum(mask)}")

    save_img(mask.astype(np.float64), 'vase_mask.png')

# -------------------------------------------------------------------------
# Challenge 1d: Compute surface normals and albedo
# -------------------------------------------------------------------------
def challenge1d():
    """Compute surface normals and albedo for the vase."""
    from computeNormals import computeNormals

    mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vase_mask.png')
    mask = np.array(Image.open(mask_path)).astype(bool)

    light_dirs_5x3 = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'light_dirs.npy'))
    vase_img_list = [load_img_uint8(f'vase{i}.png') for i in range(1, 6)]

    normals, albedo_img = computeNormals(light_dirs_5x3, vase_img_list, mask)

    # Visualize normals as a normal map image:
    # X (-1..+1) -> R (0..255), Y (-1..+1) -> G (0..255), Z (-1..+1) -> B (0..255)
    normal_map_img = ((normals + 1) / 2 * 255).astype(np.uint8)
    save_img(normal_map_img, 'vase_normal_map.png')
    save_img(albedo_img, 'vase_albedo.png')
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'normals.npy'), normals)
    print(f"  Normals shape: {normals.shape}")

# -------------------------------------------------------------------------
# Demo: Surface reconstruction (no submission required)
# -------------------------------------------------------------------------
def demoSurfaceReconstruction():
    """Demo surface reconstruction using the Frankot-Chellappa algorithm."""
    from reconstructSurf import reconstructSurf

    normals = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'normals.npy'))
    mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vase_mask.png')
    mask = np.array(Image.open(mask_path)).astype(bool)

    surf_img = reconstructSurf(normals, mask)
    save_img(surf_img, 'vase_surface.png')
    print("  Surface reconstruction complete.")

# -------------------------------------------------------------------------
# Test harness
# -------------------------------------------------------------------------
CHALLENGES = {
    'honesty': honesty,
    'challenge1a': challenge1a,
    'challenge1b': challenge1b,
    'challenge1c': challenge1c,
    'challenge1d': challenge1d,
    'demoSurfaceReconstruction': demoSurfaceReconstruction,
}

# Functions excluded from 'all' (demo functions)
DEMO_FUNCTIONS = {'demoSurfaceReconstruction'}

def run(test_name=None):
    if test_name is None or test_name == 'none':
        print("Registered challenges:")
        for name in CHALLENGES:
            print(f"  {name}")
        return

    if test_name == 'all':
        to_run = [k for k in CHALLENGES if k not in DEMO_FUNCTIONS]
    elif test_name in CHALLENGES:
        to_run = [test_name]
    else:
        print(f"***** '{test_name}' not found in registered challenges.")
        return

    for name in to_run:
        print(f"\n--- Running {name} ---")
        CHALLENGES[name]()

if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run(arg)
