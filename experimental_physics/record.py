import pyrealsense2 as rs


def record(file_name):
    input(f'About to record to {file_name}. Press ENTER to start')

    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_device('841512070547')
    config.disable_all_streams()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_record_to_file(file_name)
    pipeline.start(config)

    input(f'Recording... Press ENTER to stop')

    pipeline.stop()


if __name__ == '__main__':
    for i in range(2, 11):
        record(f'/scratch/azimov/igor/data/physics/{i:04d}.bag')
