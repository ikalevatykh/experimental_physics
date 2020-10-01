import pyrealsense2 as rs
from open3d import *
from datetime import datetime
import numpy as np

def process(file_name):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file_name)    
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    pipeline.start(config)

    pc = rs.pointcloud()
    align = rs.align(rs.stream.color)

    vis = visualization.Visualizer()
    vis.create_window('PCD', width=1280, height=720)
    pointcloud = geometry.PointCloud()
    geom_added = False

    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0])   
    vis.add_geometry(mesh_frame) 

    # box = geometry.AxisAlignedBoundingBox((-1,-1, 0), (1, 1, 0.45))

    try:
        
        while True:
            dt0=datetime.now()
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            profile = frames.get_profile()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            img_depth = geometry.Image(depth_image)
            img_color = geometry.Image(color_image)
            rgbd = geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
            
            intrinsics = profile.as_video_stream_profile().get_intrinsics()
            pinhole_camera_intrinsic = camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
            pcd = geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
            # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # print(np.asanyarray(pcd.points)[:,:,2].min(), np.asanyarray(pcd.points)[:,:,2].max())

            # pcd = pcd.crop(box)

            c = np.asanyarray(pcd.colors)

            indices = np.argwhere((c[:, 2] > 4*c[:, 1]) & (c[:, 2] > 4*c[:, 0])) #0.5*np.asanyarray(pcd.colors)[:, 1]

            pcd = pcd.select_by_index(indices)
            pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

            pcd = pcd.select_by_index(ind)

            pointcloud.points = pcd.points
            pointcloud.colors = pcd.colors


            # plane_model, inliers = pointcloud.segment_plane(distance_threshold=0.01,
            #                                         ransac_n=3,
            #                                         num_iterations=1000)

            # print(plane_model)
            
            if geom_added == False:
                vis.add_geometry(pointcloud)
                geom_added = True
            else:            
                vis.update_geometry(pointcloud)

            vis.poll_events()
            vis.update_renderer()
            
            # cv2.imshow('bgr', color_image)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break

            # process_time = datetime.now() - dt0
            # print("FPS: "+str(1/process_time.total_seconds()))

    finally:
        pipeline.stop()
        vis.destroy_window()


if __name__ == '__main__':
    process('/scratch/azimov/igor/data/physics/0001.bag')
