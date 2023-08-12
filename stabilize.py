"""
Video Stabilization Script:

This script stabilizes video by utilizing the previously computed camera shake or movement data. 
By loading the movement data from a JSON file, the script calculates the necessary transformations 
to counteract the observed camera shakes, resulting in a stabilized video. The stabilization process 
can be applied to either a single video or a directory containing multiple videos.

Before execution, ensure the paths for the video, output, and camera movement data are set correctly.
"""

import cv2
import numpy as np
import json
import os
from scipy.ndimage import gaussian_filter1d

def stabilize_video(video_path, output_path, camera_movement_data_path):
    """
    Stabilizes a video using the camera movement data.

    Args:
    - video_path: Path to the video to be stabilized
    - output_path: Path for the stabilized video output
    - camera_movement_data_path: Path to the JSON file containing camera movement data

    """
    # Load camera movement data        
    with open(camera_movement_data_path, "r") as infile:
        camera_movement_data = json.load(infile)

    # Extract dx_list and dy_list
    dx_list = [x[2] for x in camera_movement_data]
    dy_list = [x[3] for x in camera_movement_data]

    # Apply Gaussian smoothing (Optional)
    sigma = 2
    # dx_list = gaussian_filter1d(dx_list, sigma)
    # dy_list = gaussian_filter1d(dy_list, sigma)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Compute the averages for dx and dy
    dx_average = sum(dx_list) / len(dx_list)
    dy_average = sum(dy_list) / len(dy_list)

    # Subtract the averages to obtain adjusted dx and dy lists
    dx_list_adjusted = [dx - dx_list[0] for dx in dx_list]
    dy_list_adjusted = [dy - dy_list[0] for dy in dy_list]

    for i, (dx, dy) in enumerate(zip(dx_list_adjusted, dy_list_adjusted)):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Dropped frame at index {i}")
            continue

        # Compute the transformation matrix for stabilization
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        frame = cv2.warpAffine(frame, M, (width, height))

        out.write(frame)

        # Close the video feed on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

### Stabilize a single video ###
video_path = "/cicutagroup/yz655/real_videos_exp/finn_videos_original_fps/rescale_60feet_100000961607_cropped_deshaked.mp4"
output_path = "/cicutagroup/yz655/real_videos_exp/finn_videos_original_fps/stable_rescale_60feet_100000961607_cropped_deshaked.mp4" 
camera_movement_data_path = "/cicutagroup/yz655/real_videos_exp/finn_videos_original_fps/rescale_60feet_100000961607_cropped_deshaked_optical_flow_camera_movement_data.txt"
stabilize_video(video_path, output_path, camera_movement_data_path)     
  
    
###########################
# video_directory = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera_shake_detection/videos"
# output_directory = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera_shake_detection/videos"
# camera_movement_data_directory = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera_shake_detection/videos"



# for filename in os.listdir(video_directory):
#     if filename.endswith(".mp4") and filename.startswith("rescale"):
#         video_path = video_directory + '/' + filename
#         output_path = output_directory + '/' + 'stable_without_gaussian_' + filename
#         camera_movement_data_path = camera_movement_data_directory + '/' + filename.replace(".mp4", "_camera_movement_data.txt")
        
#         if os.path.exists(output_path):
#             print(f"{output_path} exists. Skipping...")
#             continue
#         else:
#             print(f"Stabilizing {video_path}...")
#             stabilize_video(video_path, output_path, camera_movement_data_path) 
    
    
### Stabilize all videos in a directory ### 

# video_directory = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera_shake_detection/drone_scale_bar_videos_rescale"
# output_directory = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera_shake_detection/drone_scale_bar_videos_stabilized"
# camera_movement_data_directory = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera_shake_detection/drone_scale_bar_videos_movement_data"



# for filename in os.listdir(video_directory):
#     if filename.endswith(".mp4") and filename.startswith("rescale_scalebar_fps10_DJI_"):
#         video_path = video_directory + '/' + filename
#         output_path = output_directory + '/' + 'stable_' + filename
#         camera_movement_data_path = camera_movement_data_directory + '/' + filename.replace("rescale_", "").replace(".mp4", "_camera_movement_data.txt")
        
#         if os.path.exists(output_path):
#             print(f"{output_path} exists. Skipping...")
#             continue
#         else:
#             print(f"Stabilizing {video_path}...")
#             stabilize_video(video_path, output_path, camera_movement_data_path) 
