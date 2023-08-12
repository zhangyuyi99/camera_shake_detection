"""
Camera Movement Detection from Videos

This script is designed to detect camera movement within videos using optical flow accerlerated by cuda. It computes the movement based on pixel changes from one frame to the next and then uses geometric computations to convert these pixel movements to real-world distances.

The main components of the script are:
1. Detection of camera movement using Farneback Optical Flow.
2. Outlier removal from the detected movements. The pixel movements caused by wheats are considered as outliers in this case.
3. Conversion of pixel movement to real-world distances.

Ensure you have the required packages installed and adjust the video path, focal length, and object distance before running the script.
"""

import cv2
import numpy as np
import math
import json
import pandas as pd
import os

def pixel_to_meter(pixel_movement, object_distance, focal_length):
    return (object_distance * pixel_movement) / focal_length

def remove_outliers(flow, threshold=1.5):
    """
    Remove outliers from the optical flow data.

    Parameters:
    - flow: Detected optical flow data.
    - threshold: Threshold value for outlier detection.

    Returns:
    - Filtered optical flow data without outliers.
    """
    median_flow_x = np.median(flow[..., 0])
    median_flow_y = np.median(flow[..., 1])

    distance = np.sqrt((flow[..., 0] - median_flow_x) ** 2 + (flow[..., 1] - median_flow_y) ** 2)
    mask = distance < threshold

    return flow[mask]

def compute_camera_movement(dx_pixels, dy_pixels, object_distance, focal_length):
    """
    Compute the camera's movement details.

    Parameters:
    - dx_pixels, dy_pixels: Pixel movement in x and y directions.
    - object_distance: Known distance to the object in meters.
    - focal_length: Camera's focal length in meters.

    Returns:
    - Tuple containing real-world movements in x and y, pixel movements in x and y, and the angle of movement.
    """
    angle = math.degrees(math.atan2(dy_pixels, dx_pixels))
    dx = pixel_to_meter(dx_pixels, object_distance, focal_length)
    dy = pixel_to_meter(dy_pixels, object_distance, focal_length)

    return dx, dy, dx_pixels, dy_pixels, angle

def main(video_path, object_distance, focal_length, output):
    """
    Main function to compute camera movement from video using optical flow.

    Parameters:
    - video_path: Path to the video file.
    - object_distance: Known distance to the object in meters.
    - focal_length: Camera's focal length in meters.
    - output: Path to save the camera movement data.
    """
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the video file")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gpu = cv2.cuda_GpuMat()
    prev_gpu.upload(prev_gray)

    camera_movement_data = []
    flow_x_list = []
    flow_y_list = []
    optical_flow = cv2.cuda_FarnebackOpticalFlow.create()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)

        flow_gpu = optical_flow.calc(prev_gpu, gray_gpu, None)
        flow = flow_gpu.download()
        
        flow_x = np.array(flow[..., 0])
        flow_y = np.array(flow[..., 1])

        # # Create a 2D histogram
        # hist, x_edges, y_edges = np.histogram2d(flow_x.flatten(), flow_y.flatten(), bins=100)

        # # Find the bin with the maximum count
        # max_bin_index = np.unravel_index(hist.argmax(), hist.shape)

        # # Get the mean values of the largest bin
        # dx_pixels = (x_edges[max_bin_index[0]] + x_edges[max_bin_index[0] + 1]) / 2
        # dy_pixels = (y_edges[max_bin_index[1]] + y_edges[max_bin_index[1] + 1]) / 2
        
        # Create a 2D histogram
        hist, x_edges, y_edges = np.histogram2d(flow_x.flatten(), flow_y.flatten(), bins=100)

        # Find the indices of the bins with the maximum 10 counts
        max_bin_indices = np.unravel_index(np.argsort(hist.ravel())[-100:], hist.shape)

        # Get the mean values of the largest bins
        dx_pixels = np.mean([x_edges[max_bin_indices[0][i]] + x_edges[max_bin_indices[0][i] + 1] for i in range(15)]) / 2
        dy_pixels = np.mean([y_edges[max_bin_indices[1][i]] + y_edges[max_bin_indices[1][i] + 1] for i in range(15)]) / 2


        # # filtered_flow = remove_outliers(flow)
        # dx_pixels = float(np.mean(flow_x))
        # dy_pixels = float(np.mean(flow_y))
        

        
        # flow_x_list.append(flow_x)
        # flow_y_list.append(flow_y)

        # Assuming distance_between_circles is known, replace it with the correct value
        distance_between_circles = 0.5
        dx, dy, dx_in_pixel, dy_in_pixel, angle = compute_camera_movement(dx_pixels, dy_pixels, object_distance, focal_length)


        camera_movement_data.append((dx, dy, dx_in_pixel, dy_in_pixel, angle))

        prev_gpu = gray_gpu

    cap.release()

    # Save camera_movement_data to a text file
    with open(output, "w") as f:
        json.dump(camera_movement_data, f)
        
    # # Convert flow_x_list and flow_y_list to NumPy arrays
    # flow_x_array = np.array(flow_x_list)
    # flow_y_array = np.array(flow_y_list)
        
    # # Save flow_x_list and flow_y_list to a text file
    # with open("optical_flow_results.txt", "w") as f:
    #     np.savetxt(f, np.column_stack((flow_x_array.flatten(), flow_y_array.flatten())), delimiter=', ', fmt='%.6f')


# object_distance = 3.8
# focal_length = 24
# video_path = "/cicutagroup/yz655/camera_shake_detection/videos/rescale_crop_fps10_DJI_0047.mp4"
# main(video_path, object_distance, focal_length)

#################################################################
# object_distance = 1
# focal_length = 1
# video_path = "/cicutagroup/yz655/real_videos_exp/finn_videos_original_fps/rescale_60feet_100000961607_cropped_deshaked.mp4"
# output = video_path[:-len(".mp4")]+"_optical_flow_camera_movement_data.txt"
# main(video_path, object_distance, focal_length, output)

################################################################

# object_distance = 1
# focal_length = 1
# video_path = "/cicutagroup/yz655/real_videos_exp/finn_videos_original_fps/rescale_30feet_100000811592_cropped_deshaked.mp4"
# output = video_path[:-len(".mp4")]+"_optical_flow_camera_movement_data.txt"
# main(video_path, object_distance, focal_length, output)

# video_folder = "/cicutagroup/yz655/wheat_videos/finn_videos/original"
# for file_name in os.listdir(video_folder):
#     video_path = os.path.join(video_folder, file_name)
#     output = "/cicutagroup/yz655/camera_shake_detection/finn_original_cmd/"+file_name[:-len(".MP4")]+"_optical_flow_camera_movement_data.txt"
#     main(video_path, object_distance, focal_length, output)
#     print(f"Video {file_name} is done.")
    


##################################################################
###
# Iterate over all drone videos from Pietro :)
###
# # Step 1: Read the Excel file
# excel_file_path = "/cicutagroup/yz655/wheat_videos/summary.xlsx"
# df = pd.read_excel(excel_file_path)

# # Step 2: Iterate over video files
# video_folder = "/cicutagroup/yz655/wheat_videos/drone_videos/drone_videos"

# for file_name in os.listdir(video_folder):
#     if file_name.endswith(".MP4") and file_name.startswith("crop_fps10") :
#         # Find the corresponding entry in the Excel file
#         video_name = file_name[len("crop_fps10_"):]
#         matching_row = df[df["FileName"] == video_name]
#         if not matching_row.empty:
#             focal_length = matching_row["FocalLength"].values[0]
#             relative_altitude = matching_row["RelativeAltitude"].values[0]
#             video_path = os.path.join(video_folder, file_name)
            
#             print(focal_length)
#             print(relative_altitude)
            
#             output = "drone_videos_optical_flow_movement_data/"+file_name[:-len(".MP4")]+"_optical_flow_camera_movement_data.txt"
#             main(video_path, relative_altitude, focal_length, output)
#             print(f"Video {video_name} is done.")
#         else:
#             print(f"Video {video_name} not found in the Excel file.")


object_distance = 1
focal_length = 1
video_folder = "/cicutagroup/yz655/wheat_videos/2023.05.30.Nottingham.UK/video"

for file_name in os.listdir(video_folder):
    if file_name.endswith(".MP4"):
        # Find the corresponding entry in the Excel file
        video_name = file_name[len("crop_fps10_"):]
        video_path = os.path.join(video_folder, file_name)
        output = video_folder+"/"+file_name[:-len(".MP4")]+"_optical_flow_camera_movement_data.txt"
        main(video_path, object_distance, focal_length, output)
        print(f"Video {video_name} is done.")
