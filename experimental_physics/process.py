import json
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import glob
import json
import os


class TrajectoryEstimator:

    def __init__(self, profile):
        self._threshold = 430

        color_stream = profile.get_stream(rs.stream.color)
        color_profile = rs.video_stream_profile(color_stream)
        self._color_intrinsics = color_profile.get_intrinsics()

        depth_stream = profile.get_stream(rs.stream.depth)
        depth_profile = rs.video_stream_profile(depth_stream)
        self._depth_intrinsics = depth_profile.get_intrinsics()
        self._depth_to_color_extrin = depth_profile.get_extrinsics_to(
            color_profile)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

    def detect(self, depth_image, color_image):
        rectangles = self._detect_rectangles(depth_image)
        if len(rectangles) == 1:
            corners = cv2.boxPoints(rectangles[0])
            x, y = np.int0(np.mean(corners, axis=0))
            depth_value = depth_image[y-5:y+5, x-5:x+5].mean()
            center = rs.rs2_deproject_pixel_to_point(
                self._depth_intrinsics, [x, y], depth_value * self._depth_scale)

            corners = [rs.rs2_deproject_pixel_to_point(
                self._depth_intrinsics, [x, y], depth_value * self._depth_scale)[:2] for x, y in corners]

            # print(corners)

            corners = np.int0(np.array(corners) * 10000)

            rect = cv2.minAreaRect(corners)
            # angle = np.deg2rad(rect)

            # angle = np.deg2rad(rectangles[0][2])
            angle = rectangles[0][2]

            cont = np.int0([self._depth_to_color([x, y], depth_value)
                            for x, y in corners])
            overlay = cv2.drawContours(color_image, [cont], 0, (0, 0, 255), 2)

            return center, angle, overlay
        return None, None, color_image

    def _detect_rectangles(self, depth_image):
        rectangles = []

        _, binary = cv2.threshold(
            depth_image, self._threshold, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            np.uint8(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            if area > 1000 and 0.8 < area / cv2.contourArea(box) < 1.2:
                rectangles.append(rect)

        return rectangles

    def _depth_to_color(self, depth_pixel, depth_value):
        depth_point = rs.rs2_deproject_pixel_to_point(
            self._depth_intrinsics, depth_pixel, depth_value * self._depth_scale)
        color_point = rs.rs2_transform_point_to_point(
            self._depth_to_color_extrin, depth_point)
        color_pixel = rs.rs2_project_point_to_pixel(
            self._color_intrinsics, color_point)
        return color_pixel


def process(file_name):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file_name, repeat_playback=False)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    estimator = TrajectoryEstimator(profile)

    times = []
    centers = []
    angles = []

    video_name = file_name.replace('.bag', '.mp4')
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (848, 480))

    try:

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image[depth_image == 0] = 1000

            center, angle, overlay = estimator.detect(depth_image, color_image)

            if center is not None and angle is not None:
                times.append(depth_frame.get_timestamp())
                centers.append(center)
                angles.append(angle)

            out.write(overlay)

            # cv2.imshow('overlay', overlay)
            # key = cv2.waitKey(50)
            # if key == ord('q'):
            #     break

    except RuntimeError:
        pass
    finally:
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()

    times = [(t - times[0]) / 1000.0 for t in times]

    for i in range(1, len(angles)):
        a0 = angles[i-1]
        a1 = angles[i]
        if a0 - a1 > 270-30:
            angles[i] += 270          
        elif a0 - a1 > 180-30:
            angles[i] += 180        
        elif a0 - a1 > 90-30:
            angles[i] += 90
        # elif a0 - a1 > 45-10:
        #     angles[i] += 45
        elif a1 - a0 > 270-30:
            angles[i] -= 270         
        elif a1 - a0 > 180-30:
            angles[i] -= 180         
        elif a1 - a0 > 90-30:
            angles[i] -= 90
        # elif a1 - a0 > 45-10:
        #     angles[i] -= 45

        # if a0 - a1 > np.pi + 0.5:
        #     angles[i] += 3 * np.pi / 4
        # elif a0 - a1 > np.pi / 2 + 0.5:
        #     angles[i] += np.pi
        # elif a0 - a1 > 0.5:
        #     angles[i] += np.pi / 2
        # elif a1 - a0 > np.pi + 0.5:
        #     angles[i] -= 3 * np.pi / 4
        # elif a1 - a0 > np.pi / 2 + 0.5:
        #     angles[i] -= np.pi
        # elif a1 - a0 > 0.5:
        #     angles[i] -= np.pi / 2

    # angles[angles > -0.1] -= np.pi / 2        

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(times, [c[0] for c in centers], label="x")
    plt.plot(times, [c[1] for c in centers], label="y")
    plt.plot(times, [c[2] for c in centers], label="z")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(times, angles, label="angle")
    plt.legend()

    plt.show()

    fig_name = file_name.replace('.bag', '.png')
    plt.savefig(fig_name)

    data = dict(time=times, center=centers, angle=np.rad2deg(angles).tolist())
    json_name = file_name.replace('.bag', '.json')
    with open(json_name, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    for name in glob.glob('/home/ikalevat/data/physics/0010.bag'):
        process(name)
    # process('/home/ikalevat/data/physics/0010.bag')
