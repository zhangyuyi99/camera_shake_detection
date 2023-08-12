"""
Camera Movement Estimation using Optical Flow

This script estimates the camera movement in a video using optical flow calculated by CPU. It calculates average motion vectors 
for each frame and then saves these motion estimates in a JSON file.

Main components:
1. Computing optical flow between consecutive frames.
2. Estimating camera motion by averaging motion vectors.
3. Saving the estimated camera movements in a JSON file.

Ensure you have the required libraries installed and set the video path appropriately.
"""

import cv2
import numpy as np
import json

def main(video_path):
    """
    Calculate average motion vectors for each frame in the video.

    Args:
    - video_path (str): Path to the video file

    Returns:
    - dx_list (list): List of average motion in the x-direction for each frame
    - dy_list (list): List of average motion in the y-direction for each frame
    """
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the video file")
        return

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    dx_list = []
    dy_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow between the previous and current frame
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Average motion vectors to estimate camera motion
        avg_motion_x = np.mean(flow[..., 0])
        avg_motion_y = np.mean(flow[..., 1])

        dx_list.append(avg_motion_x)
        dy_list.append(avg_motion_y)

        prev_frame_gray = frame_gray

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return dx_list, dy_list

def save_to_json(dx_list, dy_list, filename):
    """
    Save the camera motion data to a JSON file.

    Args:
    - dx_list (list): List of average motion in the x-direction
    - dy_list (list): List of average motion in the y-direction
    - filename (str): Name of the JSON file to save data
    """
    data = {"dx": dx_list, "dy": dy_list}
    with open(filename, "w") as file:
        json.dump(data, file)

# Path to the video file
video_path = "C:/Users/46596/Desktop/Multi DDM/data/100MEDIA/DJI_0047.MP4"
dx_list, dy_list = main(video_path)

# Name of the JSON file to save camera motion data
filename = "camera_movement_data_optical_flow.txt"
save_to_json(dx_list, dy_list, filename)
