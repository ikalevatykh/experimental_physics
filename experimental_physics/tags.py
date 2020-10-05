import pyrealsense2 as rs
import cv2
import numpy as np
import time
from pupil_apriltags import Detector
import matplotlib.pyplot as plt

from transforms3d.euler import mat2euler
import json


def test():
 
    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.5,
        debug=0,
    )

    camera_params = [425.40704345703125, 425.40704345703125, 421.0938720703125, 246.84486389160156]
    tag_size = 0.030

    centers = []
    angles = []

    for i in range(200):
        image = cv2.imread(f"/scratch/azimov/igor/data/physics/tmp/{i:04d}.png", 0)
        
        tags = at_detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size,
        )

        if len(tags) == 1:
            centers.append(tags[0].pose_t)
            euler = mat2euler(tags[0].pose_R)
            print(euler)
            angles.append(euler[2])

        # print(tags)

        print(f"{i:04d} / {200:04d}: {len(tags)}")

    for i in range(1, len(angles)):
        a0 = angles[i-1]
        a1 = angles[i]
        if a0 - a1 > 270-30:
            angles[i] += 270          
        elif a0 - a1 > 180-30:
            angles[i] += 180        
        elif a0 - a1 > 90-30:
            angles[i] += 90
        elif a1 - a0 > 270-30:
            angles[i] -= 270         
        elif a1 - a0 > 180-30:
            angles[i] -= 180         
        elif a1 - a0 > 90-30:
            angles[i] -= 90

    times = [i / 90.0 for i in rage(200)]


    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(times, [c[0] for c in centers], '-o', label="x")
    plt.plot(times, [c[1] for c in centers], '-o', label="y")
    # plt.plot([c[2] for c in centers], '-*', label="z")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(times, angles, '-o', label="angle")
    plt.legend()

    plt.show()

    data = dict(time=times, center=centers, angle=np.rad2deg(angles).tolist())
    json_name = file_name.replace('.bag', '.json')
    with open(json_name, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    test()
