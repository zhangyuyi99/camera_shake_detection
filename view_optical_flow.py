"""
Optical Flow Visualization Script:

This script reads a video file, computes the optical flow between consecutive frames, 
and visualizes the flow using hue-saturation-value (HSV) colors. The magnitude and angle 
of the flow at each pixel are encoded into the value and hue channels, respectively. 

Optical flow captures the apparent motion of objects, surfaces, and edges in a video sequence 
caused by the relative motion between the camera and the scene. It can be useful for various 
tasks such as object detection, video compression, and stabilizing videos.

Ensure the video_path is set to the input video you want to process and output_path to the desired 
location for the processed video.
"""

import cv2
import numpy as np

def visualize_optical_flow(video_path, output_path):
    """
    Visualizes optical flow of a video using the Farneback method.

    Args:
    - video_path: Path to the video for which optical flow needs to be computed
    - output_path: Path to save the visualized optical flow video
    """
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Extract video specs
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the video file")
        return

    # Convert frame to grayscale for optical flow computation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Create an HSV image to represent optical flow
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[..., 1] = 255

        # Convert flow direction and magnitude to hue and value in HSV space
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Write the optical flow visualization to output video
        out.write(bgr)
        prev_gray = gray

    # Release video resources
    cap.release()
    out.release()

# Path of the video file and output location
video_path = "/cicutagroup/yz655/camera_shake_detection/videos/rescale_crop_fps10_DJI_0047.mp4"
output_path = "output_optical_flow_DJI_0047.mp4"

# Visualize the optical flow of the given video
visualize_optical_flow(video_path, output_path)
