# Homework 2: Computer Vision, Spring 2026

## Overview

This README provides an overview of Homework #2 for CS 766: Computer Vision. The homework consists of a programming walkthrough and a programming challenge on object recognition.

## Thresholds and Combination Criteria

Walkthrough 1: 
Used number of dilations/erosions as 10 and 15 to denoise and remove rice respectively

Challenge 1a: 
The default threshold of 127 is working as expected for all the cases 

Challenge 1b:
Used the following properties:
- object label
- x coordinate of center
- y coordinate of center
- min moment
- axis/theta
- roundness

Challenge 1c:

Criteria used:
- roundness
- moment of inertia

Thresholds used:
- roundness_thresh = 0.05
- inertia_thresh = 0.20
