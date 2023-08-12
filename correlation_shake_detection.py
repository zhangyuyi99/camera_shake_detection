"""
Camera Shake Detection Using Enhanced Correlation Coefficient (ECC)

This script is designed to analyze videos and detect camera shake or movements by comparing
consecutive frames. Using the Enhanced Correlation Coefficient (ECC) algorithm, the script 
computes the transformation between frames. The detected movement, initially in pixel units, 
is then converted to real-world metrics (meters) based on a provided reference object's distance 
and the camera's focal length. The processed data is then saved to a JSON file for further analysis.

Ensure that the video_path and output_file paths are correctly set for desired file locations.
Also, providing accurate values for the object's distance and camera focal length is crucial for
reliable real-world metric computations.
"""

import cv2
import numpy as np
import math
import json

def pixel_to_meter(pixel_movement, object_distance, focal_length):
    """
    Converts pixel movement to meters based on the given object distance and the camera's focal length.

    Args:
    - pixel_movement: Movement in pixels
    - object_distance: Distance of the object in meters
    - focal_length: Camera's focal length in meters

    Returns:
    - Movement in meters
    """
    return (object_distance * pixel_movement) / focal_length

def compute_camera_movement(dx_pixels, dy_pixels, object_distance, focal_length):
    """
    Computes the camera's movement in meters and pixels and calculates the movement angle.

    Args:
    - dx_pixels: Horizontal movement in pixels
    - dy_pixels: Vertical movement in pixels
    - object_distance: Distance of the object in meters
    - focal_length: Camera's focal length in meters

    Returns:
    - dx, dy: Movement in meters
    - dx_pixels, dy_pixels: Movement in pixels
    - angle: Movement angle in degrees
    """
    angle = math.degrees(math.atan2(dy_pixels, dx_pixels))
    dx = pixel_to_meter(dx_pixels, object_distance, focal_length)
    dy = pixel_to_meter(dy_pixels, object_distance, focal_length)

    return dx, dy, dx_pixels, dy_pixels, angle

# The below version of camera_shake uses cv2.estimateAffine2D (Commented out for clarity)
# def camera_shake(video_path, object_distance, focal_length):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    camera_movement_data = []

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        transform_matrix = cv2.estimateAffine2D(prev_gray, current_gray, confidence = 0.95)

        if transform_matrix is not None:
            dx_pixels = transform_matrix[0, 2]
            dy_pixels = transform_matrix[1, 2]
            dx, dy, dx_in_pixel, dy_in_pixel, angle = compute_camera_movement(dx_pixels, dy_pixels, object_distance, focal_length)

            camera_movement_data.append((dx, dy, dx_in_pixel, dy_in_pixel, angle))

        prev_gray = current_gray

    cap.release()
    return camera_movement_data

def camera_shake(video_path, object_distance, focal_length):
    """
    Analyzes the video to detect camera shake or movements by comparing consecutive frames.

    Args:
    - video_path: Path to the video file
    - object_distance: Distance of the object in meters
    - focal_length: Camera's focal length in meters

    Returns:
    - camera_movement_data: List of movements detected (in meters, pixels, and angle)
    """
    cap = cv2.VideoCapture(video_path)

    # Ensure the video file can be read
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    ret, prev_frame = cap.read()

    # Ensure the first frame can be read
    if not ret:
        print("Error: Could not read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    camera_movement_data = []

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Define the initial transformation matrix
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Compute the transformation matrix using ECC algorithm
        _, transform_matrix = cv2.findTransformECC(prev_gray, current_gray, warp_matrix, motionType=cv2.MOTION_TRANSLATION)

        # If a transformation matrix is found, compute the camera movement
        if transform_matrix is not None:
            dx_pixels = transform_matrix[0, 2]
            dy_pixels = transform_matrix[1, 2]
            dx, dy, dx_in_pixel, dy_in_pixel, angle = compute_camera_movement(dx_pixels, dy_pixels, object_distance, focal_length)

            camera_movement_data.append((dx, dy, dx_in_pixel, dy_in_pixel, angle))

        prev_gray = current_gray

    cap.release()
    return camera_movement_data

def write_to_json(camera_movement_data, output_file):
    """
    Writes the processed camera movement data to a JSON file.

    Args:
    - camera_movement_data: List of movements detected
    - output_file: Path to the output JSON file
    """
    # Convert NumPy data types to native Python data types
    converted_camera_movement_data = [
        tuple(map(float, movement)) for movement in camera_movement_data
    ]

    with open(output_file, "w") as f:
        json.dump(converted_camera_movement_data, f)

# Main execution
if __name__ == "__main__":
    video_path = "/cicutagroup/yz655/camera_shake_detection/videos/rescale_crop_fps10_DJI_0047.mp4"
    object_distance = 3.8
    focal_length = 24
    camera_movement_data = camera_shake(video_path, object_distance, focal_length)
    
    output_file = "videos/correlation_camera_movement_data.txt"
    write_to_json(camera_movement_data, output_file)
    print(f"Camera shake data saved to {output_file}.")
