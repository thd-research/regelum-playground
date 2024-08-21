## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
# Ref: https://dev.intelrealsense.com/docs/python2?_ga=2.115289287.1346185483.1724073850-1129405603.1724073850
###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs

import cv2
from datetime import datetime
import numpy as np
import os

ROOT_DIR = "./records"
os.makedirs(ROOT_DIR, exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()


# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

video_name = datetime.now().strftime("%Y%m%d_%H%M%S_ver_1")

profile = {
    "high": (1080, 920),
    "medium": (640, 480)

}

config.enable_stream(rs.stream.color, *profile['medium'], rs.format.bgr8, 30)
# config.enable_record_to_file(os.path.join(ROOT_DIR, video_name + ".bag"))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(os.path.join(ROOT_DIR, video_name + '.avi'), fourcc, 30.0, profile['medium'])

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames(15000)
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_colormap_dim = color_image.shape

        out.write(color_image)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:
    out.release()

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
