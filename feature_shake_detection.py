"""
Camera Movement Detection by Scale Bar Tracking

This script is designed to detect the movement of a drone camera 
by tracking a known stationary scale bar placed in a wheat field by
 utilizing computer vision and tracking techniques.

Note: Ensure the stationary scale bar is placed well within the 
frame for accurate tracking.
"""

import cv2
import numpy as np
import json
import math
import os

def select_roi(frame, title, max_width=600, max_height=600):
    """
    Allows the user to select a region of interest (ROI) on a given frame.

    Args:
    - frame: Input frame.
    - title: Window title for the selection interface.
    - max_width, max_height: Maximum dimensions for resizing.

    Returns:
    - A tuple containing coordinates (x, y, width, height) of the selected ROI.
    """
    
    # Calculate the scaling factor
    img_height, img_width, _ = frame.shape
    scale = min(max_width / img_width, max_height / img_height)
    
    # Resize the frame for easier ROI selection
    frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
    
    # Let the user select the ROI
    roi_resized = cv2.selectROI(title, frame_resized, False, False)
    cv2.destroyWindow(title)
    
    # Convert the ROI coordinates back to the original frame size
    x, y, w, h = [int(v / scale) for v in roi_resized]
    
    return x, y, w, h


def detect_circle(frame, roi):
    """
    Detect circles within a given ROI in a frame.

    Args:
    - frame: Input frame.
    - roi: Region of interest tuple (x, y, width, height).

    Returns:
    - Tuple containing center and radius of detected circle or None if no circle detected.
    """
    
    # Extract the region of interest from the frame
    x, y, w, h = roi
    cropped = frame[y:y+h, x:x+w]
    
    # Convert the cropped frame to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # Extract edges from the grayscale image
    edges = cv2.Canny(gray, 100, 200)
    
    # Detect circles using the HoughCircles method
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=10,
                                param1=100, param2=10, minRadius=0, maxRadius=0)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circle = circles[0]
        circle[0] += x
        circle[1] += y
        return tuple(circle)
    else:
        return None

def compute_camera_movement(p1, p2, distance_between_circles):
    """
    Compute the camera's movement based on the tracked points.

    Args:
    - p1, p2: Tuple coordinates of the tracked points.
    - distance_between_circles: Known distance between the circles in meters.

    Returns:
    - Tuple containing computed displacements and angle of movement.
    """
    
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    dx_pixels = x2 - x1
    dy_pixels = y2 - y1
    
    # Calculate the displacement in pixels
    distance_pixels = math.sqrt(dx_pixels ** 2 + dy_pixels ** 2)
    scale = distance_between_circles / distance_pixels
    
    dx_in_pixel = (x1+x2) / 2
    dy_in_pixel = (y1+y2) / 2   
    dx = (x1+x2) * scale / 2
    dy = (y1+y2) * scale / 2
    
    # Calculate the rotation angle
    angle = math.degrees(math.atan2(dy_pixels, dx_pixels))
    
    return dx, dy, dx_in_pixel, dy_in_pixel, angle

def main(video_path, distance_between_circles, output_filename, show_scale):
    """
    Main function to process the video, detect circles, and track them.

    Args:
    - video_path: Path to the input video.
    - distance_between_circles: Known distance between circles in meters.
    - output_filename: Output file to store the computed camera movement data.
    - show_scale: Dimension for displaying the processed video frame.

    Returns:
    - List containing camera movement data.
    """
    
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Error reading the video file")
        return
    
    # Allow the user to select two circles
    roi1 = select_roi(frame, "Select Circle 1", show_scale, show_scale)
    roi2 = select_roi(frame, "Select Circle 2", show_scale, show_scale)

    circle1 = detect_circle(frame, roi1)
    circle2 = detect_circle(frame, roi2)
    
    camera_movement_data = []

    if circle1 is not None and circle2 is not None:
        # Initialize trackers for the detected circles
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        tracker1 = cv2.TrackerKCF_create()
        tracker2 = cv2.TrackerKCF_create()

        bbox1 = (x1 - r1, y1 - r1, 2 * r1, 2 * r1)
        bbox2 = (x2 - r2, y2 - r2, 2 * r2, 2 * r2)

        tracker1.init(frame, bbox1)
        tracker2.init(frame, bbox2)

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ret1, bbox1 = tracker1.update(frame)
            ret2, bbox2 = tracker2.update(frame)

            if ret1 and ret2:
                # Extract the center coordinates from the bounding boxes
                x1, y1, w1, h1 = bbox1
                x2, y2, w2, h2 = bbox2

                p1 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                p2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))

                cv2.circle(frame, p1, 5, (0, 255, 0), -1)
                cv2.circle(frame, p2, 5, (0, 0, 255), -1)

                dx, dy, dx_in_pixel, dy_in_pixel, angle = compute_camera_movement(p1, p2, distance_between_circles)

                camera_movement_data.append({
                    "frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
                    "dx": dx,
                    "dy": dy,
                    "dx_in_pixel": dx_in_pixel,
                    "dy_in_pixel": dy_in_pixel,
                    "angle": angle
                })

            # Display the processed frame
            cv2.imshow("Camera Movement Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save the camera movement data to a JSON file
        with open(output_filename, 'w') as f:
            json.dump(camera_movement_data, f)

    else:
        print("Failed to detect circles!")

    return camera_movement_data

# Configurable parameters
VIDEO_PATH = "path_to_video.mp4"
DISTANCE_BETWEEN_CIRCLES = 1.0  # in meters
OUTPUT_FILENAME = "camera_movement_data.json"
SHOW_SCALE = 600

# Call the main function
camera_movement_data = main(VIDEO_PATH, DISTANCE_BETWEEN_CIRCLES, OUTPUT_FILENAME, SHOW_SCALE)
print(camera_movement_data)


video_directory = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera_shake_detection/drone_scale_bar_videos"
output_directory = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera_shake_detection/drone_scale_bar_videos_movement_data"

distance_between_circles = 1.0  # meters
show_scale = 600
# focal_length_mm = 12.0
# altitude = 3.8 # meters

# filename="rescale_scalebar_fps10_DJI_0048.mp4"


# if filename.endswith(".mp4") and filename.startswith("rescale_scalebar_fps10_DJI_"):
#     video_path = video_directory + '/' + filename
#     output_path = output_directory + '/' + filename[:-4]+'_camera_movement_data.txt'
#     # if os.path.exists(output_path):
#     #     print(f"{output_path} exists. Skipping...")
#     # else:
#     #     print(f"Processing {video_path}...")
#     #     main(video_path, distance_between_circles, output_path)
#     main(video_path, distance_between_circles, output_path, show_scale)

# Iterate through all video files in the video_directory
for filename in os.listdir(video_directory):
    if filename.endswith(".mp4") and filename.startswith("scalebar_fps10_DJI_"):
        video_path = video_directory + '/' + filename
        output_path = output_directory + '/' + filename[:-4]+'_camera_movement_data.txt'
        if os.path.exists(output_path):
            print(f"{output_path} exists. Skipping...")
            continue
        else:
            print(f"Processing {video_path}...")
            main(video_path, distance_between_circles, output_path, show_scale)

# video_path = "C:/Users/46596/Desktop/Multi DDM/data/100MEDIA/DJI_0047.MP4"


# video_path = "//sf3.bss.phy.private.cam.ac.uk/cicutagroup/yz655/camera shake detection/shake_videos/rescale_crop_fps10_DJI_0047.MP4"
# output_filename = "rescale_crop_fps10_DJI_0047_camera_movement_data.txt"


# camera_movement_data = main(video_path, distance_between_circles, output_filename)



### dx and dy: These represent the horizontal and vertical displacements of the camera, respectively. Their units are in meters (m) since the altitude is given in meters and the distance between the circles is also in meters.
### dx_in_pixel and dy_in_pixel: displacement in pixels, used for video stabilization
### angle: This represents the camera's rotating angle. The unit for this value is in degrees (°), as the math.degrees() function is used to convert the angle from radians to degrees.
