import os
import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from utils.geometry_utils import load_transformations
from utils.geometry_utils import accumulate_absolute_transformations
from utils.geometry_utils import accumulate_relative_transformations

def plot_transformed_frames(est_poses, R, t, s, color):
    frames = []
    for pose in est_poses:
        # Applica la trasformazione a ciascuna posa stimata
        transformed_pose = apply_transformation_to_pose(pose, R, t, s)
        
        # Crea il frame di riferimento su Open3D
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(transformed_pose)
        frames.append(frame)
    return frames

def visualize_absolute_frames(gt_camera_folder, est_camera_folder, color_flag, size=1000):
    # We assume the files are named in the format 0000.yaml, 0001.yaml
    gt_cam_files = {int(f.split('.')[0]) for f in sorted(os.listdir(gt_camera_folder)) if f.endswith(".yaml")}
    est_cam_files = {int(f.split('.')[0]) for f in sorted(os.listdir(est_camera_folder)) if f.endswith(".yaml")}
    # Common indices between the two folders
    common_indices = gt_cam_files & est_cam_files  

    # Extract relative transformations
    gt_transformations, est_transformations = load_transformations(gt_camera_folder, est_camera_folder, common_indices)

    # Extract poses with respect to the first pose
    gt_accumulated = accumulate_absolute_transformations(gt_transformations[:size])
    est_accumulated = accumulate_absolute_transformations(est_transformations[:size])

    print(f"Loaded {len(gt_accumulated)} ground truth poses and {len(est_accumulated)} estimated poses.")

    gt_color = [0, 1, 0]  # Green for ground truth
    est_color = [1, 0, 0]  # Red for estimated poses

    # Let's create all the reference frames
    gt_frames = plot_camera_frames(gt_accumulated, gt_color, color_flag)
    est_frames = plot_camera_frames(est_accumulated, est_color, color_flag)

    # Let's visualize them
    o3d.visualization.draw_geometries(gt_frames + est_frames)

    return gt_accumulated, est_accumulated, common_indices

def visualize_relative_frames(gt_camera_folder, est_camera_folder, color_flag, size=1000):
    # We assume the files are named in the format 0000.yaml, 0001.yaml
    gt_cam_files = {int(f.split('.')[0]) for f in sorted(os.listdir(gt_camera_folder)) if f.endswith(".yaml")}
    est_cam_files = {int(f.split('.')[0]) for f in sorted(os.listdir(est_camera_folder)) if f.endswith(".yaml")}

    # Common indices between the two folders
    common_indices = gt_cam_files & est_cam_files  

    # Extract relative transformations
    gt_transformations, est_transformations = load_transformations(gt_camera_folder, est_camera_folder, common_indices)

    # Extract poses with respect to the first pose
    gt_accumulated = accumulate_relative_transformations(gt_transformations[:size])
    est_accumulated = accumulate_relative_transformations(est_transformations[:size])

    print(f"Loaded {len(gt_accumulated)} ground truth poses and {len(est_accumulated)} estimated poses.")

    gt_color = [0, 1, 0]    # Green for ground truth
    est_color = [1, 0, 0]   # Red for estimated poses

    # Let's create all the reference frames
    gt_frames = plot_camera_frames(gt_accumulated, gt_color, color_flag)
    est_frames = plot_camera_frames(est_accumulated, est_color, color_flag)

    # Let's visualize them
    o3d.visualization.draw_geometries(gt_frames + est_frames)

    return gt_accumulated, est_accumulated, common_indices

def plot_camera_frames(poses, color, color_flag=True, size=0.05):
    frames = []
    for pose in poses:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(pose)
        if color_flag:
            frame.paint_uniform_color(color)
        frames.append(frame)

        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Sphere size proportional to frame size
        center.translate(pose[:3, 3])  # Position the sphere at the origin of the pose
        center.paint_uniform_color(color)  # Set color of the center point
        frames.append(center)

    return frames

def save_scaled_poses(input_dir, relative_poses, common_indices):
    # Save each relative transformation as a YAML file in OpenCV format
    output_camera_poses_dir = os.path.join(input_dir, "scaled_estimated_camera_poses")
    os.makedirs(output_camera_poses_dir, exist_ok=True)
    for i, T_rel in enumerate(relative_poses):
        # Define the output file name
        filename = os.path.join(output_camera_poses_dir, f"{common_indices[i]:04}.yaml")
        # Open a cv2.FileStorage for writing in YAML format
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("matrix", T_rel)
        fs.release()

        print(f"Saved relative transformation to {filename}")

def show_frames_sequentially(gt_frames, aligned_est_frames, aligned_est_poses, gt_accumulated):
    # Check that the number of frames matches
    if len(gt_frames) != len(aligned_est_frames):
        print("Error: Ground truth and estimated frames lists have different lengths.")
        return

    # Create a visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    

    # Initialize a variable to keep track of the current frame index
    current_frame = 0
    def update_geometry(vis):
        nonlocal current_frame

        # Clear any existing geometries
        vis.clear_geometries()
        reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)  # Dimensione grande per visibilità
        vis.add_geometry(reference_frame)
        # Add the current pair of frames to the visualizer
        vis.add_geometry(gt_frames[current_frame])
        vis.add_geometry(aligned_est_frames[current_frame])
         
        print("------ current_frame: ", current_frame, "------")
        camera_pose = np.array([aligned_est_poses[current_frame][0, 3], aligned_est_poses[current_frame][1, 3], aligned_est_poses[current_frame][2, 3]])
        print("Aligned est frame: ", camera_pose)
        robot_pose = gt_accumulated[current_frame][:3, 3]
        print("Initial gt frame: ", robot_pose)
        rob_distances = np.linalg.norm(robot_pose, axis=0)
        print("Robot distances: ", rob_distances)
        
        cam_distances = np.linalg.norm(camera_pose, axis=0)
        print("Camera distances: ", cam_distances)
        camera_rotation_matrix = aligned_est_poses[current_frame][:3, :3]
        robot_rotation_matrix = gt_accumulated[current_frame][:3, :3]

        camera_yaw = np.arctan2(camera_rotation_matrix[1, 0], camera_rotation_matrix[0, 0])
        robot_yaw = np.arctan2(robot_rotation_matrix[1, 0], robot_rotation_matrix[0, 0])
        z_axis_rotation_difference = np.rad2deg(camera_yaw - robot_yaw)

        print("Rotation difference around Z-axis (degrees): ", abs(z_axis_rotation_difference))
        scale_factors = rob_distances / cam_distances
        print("Scale factors: ", scale_factors)

        # Increment to the next frame, looping if at the end
        current_frame = (current_frame + 1) % len(gt_frames)
        return False  # Continue running the visualizer

    # Register the space bar (key code 32) to advance frames
    vis.register_key_callback(32, update_geometry)
    
    # Start by adding the first pair
    vis.add_geometry(gt_frames[0])
    vis.add_geometry(aligned_est_frames[0])
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()



def save_corrected_robot_poses(input_dir, relative_poses):
    # Save each relative transformation as a YAML file in OpenCV format
    output_camera_poses_dir = os.path.join(input_dir, "corrected_robot_poses")
    os.makedirs(output_camera_poses_dir, exist_ok=True)
    for i, T_rel in enumerate(relative_poses):
        # Define the output file name
        filename = os.path.join(output_camera_poses_dir, f"{i:04}.yaml")
        # Open a cv2.FileStorage for writing in YAML format
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("matrix", T_rel)
        fs.release()

        print(f"Saved relative transformation to {filename}")