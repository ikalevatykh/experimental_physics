import json

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
from transforms3d.euler import mat2euler


class Estimator:

    def __init__(self, intrinsics, tag_size):
        self._detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.5,
            debug=0,
        )

        self._camera_params = [
            intrinsics.fx, intrinsics.fy,
            intrinsics.ppx, intrinsics.ppy,
        ]
        self._tag_size = tag_size

    def cube_pose(self, image):
        tags = self._detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=self._camera_params,
            tag_size=self._tag_size,
        )

        if len(tags) == 1:
            euler = mat2euler(tags[0].pose_R)
            return tags[0].pose_t, euler[2]

        return None


def process(file_name):
    print(f'Process {file_name}')

    dataset = []
    failures = 0

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file_name, repeat_playback=False)
    config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
    profile = pipeline.start(config)

    try:
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        ir_stream = profile.get_stream(rs.stream.infrared)
        ir_profile = rs.video_stream_profile(ir_stream)
        intrinsics = ir_profile.get_intrinsics()

        estimator = Estimator(intrinsics, tag_size=0.030)

        while True:
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            ir_image = np.asanyarray(ir_frame.get_data())
            pose = estimator.cube_pose(ir_image)

            if pose is not None:
                dataset.append((ir_frame.get_timestamp(), pose))
            else:
                failures += 1
    except RuntimeError:
        pass
    finally:
        pipeline.stop()

    if failures:
        print(f'{failures} failures')

    dump(dataset, file_name, show=False)


def dump(dataset, file_name, show=False):
    timestamps = [t for t, _ in dataset]
    timestamps = [(t - timestamps[0]) / 1000.0 for t in timestamps]
    positions = [pose[0].tolist() for _, pose in dataset]
    orientations = consistent([pose[1] for _, pose in dataset])

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, [x for x, y, z in positions], '-o', label="x")
    plt.plot(timestamps, [y for x, y, z in positions], '-o', label="y")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, orientations, '-o', label="angle")
    plt.legend()

    if show:
        plt.show()

    fig_name = file_name.replace('.bag', '.png')
    plt.savefig(fig_name)

    data = dict(time=timestamps, center=positions, angle=orientations)
    json_name = file_name.replace('.bag', '.json')
    with open(json_name, 'w') as f:
        json.dump(data, f)


def consistent(angles):
    for i in range(1, len(angles)):
        a0 = angles[i-1]
        a1 = angles[i]
        for q in [2*np.pi, 3*np.pi/4, np.pi, np.pi/2]:
            if a0 - a1 > q-0.5:
                angles[i] += q
            elif a1 - a0 > q-0.5:
                angles[i] -= q
    return angles


if __name__ == "__main__":
    for i in range(11, 21):
        process(f'/home/ikalevat/data/physics/{i:04d}.bag')
