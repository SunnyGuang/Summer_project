# using opencv via realsense d435i detect the real-time skeleton

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Make detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        color_image.flags.writeable = False
        results = pose.process(color_image)

        # Draw the pose annotation on the image.
        color_image.flags.writeable = True
        mp_drawing.draw_landmarks(
            color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', color_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cv2.destroyAllWindows()
pipeline.stop()
