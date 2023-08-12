import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Progressbar, Label, Entry, Button
from threading import Thread
import cv2
import numpy as np
import math
import json
import os

class CameraMovementProcessor:
    def __init__(self):
        pass

    @staticmethod
    def pixel_to_meter(pixel_movement, object_distance, focal_length):
        return (object_distance * pixel_movement) / focal_length

    @staticmethod
    def compute_camera_movement(dx_pixels, dy_pixels, object_distance, focal_length):
        angle = math.degrees(math.atan2(dy_pixels, dx_pixels))
        dx = CameraMovementProcessor.pixel_to_meter(dx_pixels, object_distance, focal_length)
        dy = CameraMovementProcessor.pixel_to_meter(dy_pixels, object_distance, focal_length)

        return dx, dy, dx_pixels, dy_pixels, angle

    @staticmethod
    def process_video(video_file, object_distance, focal_length, output):
        cap = cv2.VideoCapture(video_file)

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

            hist, x_edges, y_edges = np.histogram2d(flow_x.flatten(), flow_y.flatten(), bins=100)

            max_bin_indices = np.unravel_index(np.argsort(hist.ravel())[-100:], hist.shape)

            dx_pixels = np.mean([x_edges[max_bin_indices[0][i]] + x_edges[max_bin_indices[0][i] + 1] for i in range(15)]) / 2
            dy_pixels = np.mean([y_edges[max_bin_indices[1][i]] + y_edges[max_bin_indices[1][i] + 1] for i in range(15)]) / 2

            dx, dy, dx_in_pixel, dy_in_pixel, angle = CameraMovementProcessor.compute_camera_movement(dx_pixels, dy_pixels, object_distance, focal_length)

            camera_movement_data.append((dx, dy, dx_in_pixel, dy_in_pixel, angle))

            prev_gpu = gray_gpu

        cap.release()

        # Save camera_movement_data to a text file
        with open(output, "w") as f:
            json.dump(camera_movement_data, f)

        print(f"Processing complete for video: {video_file}")


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid(sticky="nesw")
        self.create_widgets()

        # Default output directory
        self.output_directory = os.getcwd()

    def create_widgets(self):
        Label(self, text="Object Distance:").grid(row=0, column=0)
        self.object_distance_entry = Entry(self)
        self.object_distance_entry.grid(row=0, column=1)

        Label(self, text="Focal Length:").grid(row=1, column=0)
        self.focal_length_entry = Entry(self)
        self.focal_length_entry.grid(row=1, column=1)

        select_video_button = Button(self, text="Select Videos", command=self.select_video)
        select_video_button.grid(row=2, column=0)

        self.video_label = Label(self, text="")
        self.video_label.grid(row=2, column=1)

        select_output_button = Button(self, text="Select Output Directory", command=self.select_output_directory)
        select_output_button.grid(row=3, column=0)

        self.output_label = Label(self, text="Output: Current Directory")
        self.output_label.grid(row=3, column=1)

        start_processing_button = Button(self, text="Start Processing", command=self.start_processing)
        start_processing_button.grid(row=4, column=0, columnspan=2)

        self.progress = Progressbar(self, length=100, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=2)

    def select_video(self):
        self.video_files = filedialog.askopenfilenames(filetypes=(("Video files", "*.mp4;*.avi"), ("All files", "*.*")))
        self.video_label['text'] = f'Selected {len(self.video_files)} video(s)'

    def select_output_directory(self):
        self.output_directory = filedialog.askdirectory()
        self.output_label['text'] = f"Output: {self.output_directory}"

    def start_processing(self):
        object_distance = float(self.object_distance_entry.get())
        focal_length = float(self.focal_length_entry.get())
        self.progress.start()
        self.thread = Thread(target=self.process_videos, args=(object_distance, focal_length,))
        self.thread.start()
        self.after(20, self.check_thread)

    def process_videos(self, object_distance, focal_length):
        for video_file in self.video_files:
            output = os.path.join(self.output_directory, os.path.basename(video_file)[:-4]+"_optical_flow_camera_movement_data.txt")
            CameraMovementProcessor.process_video(video_file, object_distance, focal_length, output)
        self.progress.stop()
        messagebox.showinfo("Info", "Processing complete")

    def check_thread(self):
        if self.thread.is_alive():
            self.after(20, self.check_thread)
        else:
            messagebox.showinfo("Info", "Processing complete")


root = tk.Tk()
root.geometry('400x200')
app = Application(master=root)
app.mainloop()
