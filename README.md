# Camera Shake Detection for Drones in Wheat Fields

This project is dedicated to detecting the shakiness of cameras mounted on drones when capturing videos of wheat fields from a top-down perspective. One of the challenges in this task is that there's often no distinct object to track. 

## Overview

We employ several methods to detect and quantify the camera shake:

- When there's a scale bar placed within the field, we track the bar using computer vision methods (`feature_shake_detection.py`).
  
- In the absence of a scale bar, we utilize optical flow and correlation techniques for shake detection.
  
By comparing the three primary camera shake detection methods in `compare_stable_methods.ipynb`, we've found that they align well with one another. With the acquired camera shake data, we can stabilize the video (`stabilize.py`). Evaluating the quality and effectiveness of this stabilization is done in `stabilize_quality.ipynb`.

One of the key findings from this work is its implication on Differential Dynamic Microscopy (DDM) analysis. By reducing camera shakiness, we significantly decrease noise in DDM analysis, which further aids in the accurate recovery of wave information. This is illustrated in `camera_shake_plot_directioned_Iqtau.mlx`.

## Project Files

### Camera Shake Detection

- `feature_shake_detection.py` - Detects camera shake by tracking features (e.g., scale bar).

- `correlation_shake_detection.py` - Script for camera shake detection using the Enhanced Correlation Coefficient (ECC).

- `optical_flow_homography_shake_detection_cuda.py` - Optical flow-based shake detection using CUDA acceleration.

- `optical_flow_shake_detection.py` & `optical_flow_shake_detection_cuda.py` - Scripts for shake detection using optical flow, with the latter benefiting from CUDA acceleration.

- `shake_detection_gui.py` - A GUI for real-time shake detection.

- `optical_flow_remove_outliers.ipynb` - A notebook to view outliers in optical flow analysis.

### Video Stabilizing

- `stabilize.py` - Script to stabilize the video based on detected camera shake data.

- `stabilize_quality.ipynb` - A notebook evaluating the quality of video stabilization.

- `view_optical_flow.py` - Script to visualize optical flow of video pixels.

### Data Analysis and Plotting

- `camera_shake_plot_directioned_Iqtau.mlx` - DDM analysis results after camera shake removal.

- `camera_shake_wind_analysis.mlx` - Analyzing the influence of wind on camera shakiness.

- `compare_stable_methods.ipynb` - A notebook comparing the outcomes of different shake detection methods.


## Conclusions

By accurately detecting and subsequently reducing the camera shakiness, we enhance the clarity and quality of drone-captured videos in wheat fields. This not only improves the visual appeal but significantly aids in analytical tasks such as DDM, providing more accurate wave information.

