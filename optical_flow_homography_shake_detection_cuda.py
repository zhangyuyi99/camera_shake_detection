import cv2
import numpy as np
import math
import json
import pandas as pd
import os
from itertools import islice

def pixel_to_meter(pixel_movement, object_distance, focal_length):
    return (object_distance * pixel_movement) / focal_length

def batch(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # from itertools recipes
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk

def select_solution(rotations, translations):
    min_rotation_magnitude = float('inf')
    best_rotation = None
    best_translation = None

    for rotation, translation in zip(rotations, translations):
        # calculate the magnitude of rotation around X and Y axis
        rotation_magnitude = np.sqrt(rotation[2][0]**2 + rotation[2][1]**2)

        # if this is the smallest we've seen, update our best rotation and translation
        if rotation_magnitude < min_rotation_magnitude:
            min_rotation_magnitude = rotation_magnitude
            best_rotation = rotation
            best_translation = translation

    return best_rotation, best_translation

def main(video_path, object_distance, focal_length, output, frame_skip=1):
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

    H_prev = None
    
    # Create an iterator for the frames
    frames = iter(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    # while cap.isOpened():
        
    for _ in batch(frames, frame_skip):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)

        flow_gpu = optical_flow.calc(prev_gpu, gray_gpu, None)
        flow = flow_gpu.download()

        # create a grid of points and track them using optical flow
        grid = np.indices(prev_gray.shape).reshape(2, -1).T.astype(np.float32)
        next_grid = grid + flow.reshape(2, -1).T

        # compute homography from the tracked points
        H, status = cv2.findHomography(grid, next_grid, cv2.RANSAC)

        # If no good match is found for the current frame or if H is None, use the last calculated homography matrix
        if status is None or H is None:
            H = H_prev
        else:
            H_prev = H

        # Assume image shape is (height, width)
        height, width = prev_gray.shape[:2]
        K = np.array([[width, 0, width/2],
                    [0, height, height/2],
                    [0, 0, 1]], dtype=np.float32)

        # Then pass K to cv2.decomposeHomographyMat()
        num, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
        
        # select the solution satisfying the constraints of camera motion
        rotation, translation = select_solution(rotations, translations)

        camera_movement_data.append((translation[0], translation[1], rotation[0], rotation[1]))

        prev_gpu = gray_gpu

    cap.release()

    # When saving the camera_movement_data, convert numpy arrays to lists
    with open(output, "w") as f:
        # camera_movement_data_list = [(translation[0].tolist(), translation[1].tolist(), rotation[0].tolist(), rotation[1].tolist()) for translation, rotation in camera_movement_data]
        # json.dump(camera_movement_data_list, f)
        json.dump(camera_movement_data, f)


# Adjust parameters accordingly
video_path = "/path/to/video"
output = "/path/to/output"
video_path = "/cicutagroup/yz655/camera_shake_detection/videos/rescale_crop_fps10_DJI_0047.mp4"
output = video_path[:-len(".mp4")]+"_optical_flow_homography_camera_movement_data.txt"
object_distance = 3.8
focal_length = 24
main(video_path, object_distance, focal_length, output, frame_skip=2)
