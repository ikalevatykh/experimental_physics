import pyrealsense2 as rs
import cv2
import numpy as np
import time


def record(file_name):
    input(f"About to record to {file_name}. Press ENTER to start")

    context = rs.context()
    devices = context.query_devices()
    for dev in devices:
        sensors = dev.query_sensors()
        for sensor in sensors:
            if sensor.is_depth_sensor():
                sensor.set_option(rs.option.exposure, 200.0)
                sensor.set_option(rs.option.gain, 200.0)
                sensor.set_option(rs.option.emitter_enabled, 0)
                sensor.set_option(rs.option.enable_auto_exposure, 0)

    config = rs.config()
    config.disable_all_streams()
    config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
    config.enable_record_to_file(file_name)

    pipeline = rs.pipeline()
    pipeline.start(config)
    try:
        input(f"Recording... Press ENTER to stop")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    for i in range(20, 21):
        record(f"/scratch/azimov/igor/data/physics/{i:04d}.bag")
