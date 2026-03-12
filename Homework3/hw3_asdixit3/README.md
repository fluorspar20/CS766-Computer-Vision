# CS766 Homework 3: Hough Transform

This document provides an overview of the various design choices taken while working on homework 3 for CS 766. The assignment consists of a programming walkthrough and a programming challenge, focusing on image processing and edge detection techniques using Hough Transform.

## Design decisions, Voting Scheme and Algorithm

#### Challenge 1a
Used canny edge detection technique over sobel, got better results.

#### Challenge 1b
Voting is done using the simple line detection algorithm: for every nonzero edge pixel we loop over `theta_num_bins` values of \(\theta\in[0,\pi)\) and compute \(\rho = y_i\cos\theta - x_i\sin\theta\) with the origin placed at the centre of the image.  The resulting \(\rho\) is quantized to the nearest of `rho_num_bins` bins and the corresponding cell in `hough_img` is incremented.  A resolution of 720 rho bins by 360 theta bins was chosen so that the parameters are fine enough to locate lines accurately without making the votes insignificant.  After voting the accumulator is normalised to the range 0–255.

#### Challenge 1c
Peak detection starts by thresholding the accumulator with the supplied `hough_threshold` value (100, 60 and 110 for the three images).  Every bin above that value is considered a candidate. We then apply non‑maximum suppression using a 5×5 neighbourhood. The resulting (rho, theta) peaks are converted back to line equations and drawn in red on the original image.

#### Challenge 1d
To prune infinite lines down to actual segments the code in `line_segment_finder` performs the following steps for each NMS peak (same 5×5 neighbourhood):

1.  Compute a Canny edge map of the original image with `sigma=2.0`, `low_threshold=0.04*np.max(gray_img)` and `high_threshold=0.08*np.max(gray_img)`.
2.  Translate the edge coordinates to the same centre‑based coordinate system and compute the signed distance from each edge point to the infinite line. Points whose distance is less than 2 pixels are treated as inliers.
3.  Project the inliers onto the line direction and sort them, then split the sorted list wherever successive projections differ by more than 10 pixels (this breaks lines at large gaps caused by missing edges).
4.  Discard any resulting segment shorter than 35 pixels, and draw the remaining segment’s two end‑points on the image.

