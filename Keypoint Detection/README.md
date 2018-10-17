# Keypoint Detection

This repository addresses the implementation of keypoint detection and illustrates a program to detect keypoints in an image according to the following steps, which are also the first three steps of Scale-Invariant Feature Transform (SIFT).

Step 1: Generate four octaves. Each octave is composed of five images blurred using Gaussian kernels.

Step 2: Compute Difference of Gaussian (DoG) for all four octaves.

Step 3: Detect keypoints which are located at the maxima or minima of the DoG images.
