#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import math
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import torch
import cv2
import open3d as o3d

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

from mast3r.utils.general_utils import generate_mask_list
from Grounded_SAM_2.sam2_mask_tracking import MaskGenerator 

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_single_masks, load_single_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser
from mast3r.utils.general_utils import rotation_matrix_to_rpy, compute_errors, read_transformations
import matplotlib.pyplot as pl


from scipy.spatial import cKDTree

def compute_chessboard_square_sizes(pts3d, pattern_size):
    """
    Calcola le distanze tra corner adiacenti in 3D (orizzontali e verticali).

    Args:
        pts3d: (N, 3) array dei corner 3D (ordinati da OpenCV)
        pattern_size: (cols, rows), numero di corner interni (es: (9, 6))

    Returns:
        horizontal_distances: distanze tra corner orizzontali
        vertical_distances: distanze tra corner verticali
    """
    cols, rows = pattern_size
    pts3d = np.array(pts3d)
    if pts3d.shape[0] != cols * rows:
        print("⚠️ Numero di corner 3D insufficiente per pattern completo.")
        return None, None

    pts3d_grid = pts3d.reshape((rows, cols, 3))  # shape (rows, cols, 3)

    horiz_dists = []
    vert_dists = []

    for r in range(rows):
        for c in range(cols - 1):
            d = np.linalg.norm(pts3d_grid[r, c + 1] - pts3d_grid[r, c])
            horiz_dists.append(d)

    for r in range(rows - 1):
        for c in range(cols):
            d = np.linalg.norm(pts3d_grid[r + 1, c] - pts3d_grid[r, c])
            vert_dists.append(d)

    return np.array(horiz_dists), np.array(vert_dists)

def extract_3d_chessboard_corners(imgs, camera_poses, K, pcd_path, pattern_size=(7, 6), max_pixel_error=1):
    """
    Estrae i corner della scacchiera da immagini e trova i punti 3D corrispondenti nella pointcloud.

    Args:
        imgs: lista di immagini RGB in float32 (valori [0,1])
        camera_poses: lista di (4x4) numpy array, uno per immagine (T_c2w)
        K: matrice intrinseca 3x3
        pointcloud_path: path a un file .ply con la nuvola di punti
        pattern_size: dimensione griglia scacchiera (cols, rows)
        max_pixel_error: distanza massima in pixel per considerare un match valido

    Returns:
        lista di (Nx2, Nx3): corner 2D e corrispondenti 3D per ogni immagine
    """
    # Carica point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    points_world = np.asarray(pcd.points)  # Nx3

    results = []

    output_dir = "output_corners"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(imgs):
        img_rgb = (img * 255).astype(np.uint8)
        img_vis = img_rgb.copy()

        # Converti immagine in uint8
        img_gray = (img * 255).astype(np.uint8)
        if img_gray.shape[2] == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

        # Trova corner
        # found, corners = cv2.findChessboardCorners(img_gray, pattern_size)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(img_gray, pattern_size, flags)
        
        if not found:
            results.append((None, None))
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
        
        img_with_corners = cv2.drawChessboardCorners(img_vis.copy(), pattern_size, corners, found)

        # Salva l’immagine con i corner
        outpath = os.path.join(output_dir, f"image_{i:03d}_corners.png")
        cv2.imwrite(outpath, cv2.cvtColor(img_with_corners, cv2.COLOR_RGB2BGR))

        corners = corners.squeeze(1)  # (N, 2)

        # Pose: trasformazione camera -> mondo
        T_c2w = camera_poses[i]
        R_wc = T_c2w[:3, :3]
        t_wc = T_c2w[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        # Trasforma la pointcloud nel frame della camera
        points_cam = (R_cw @ points_world.T + t_cw[:, np.newaxis]).T

        # Filtra solo i punti davanti alla camera
        valid_idx = points_cam[:, 2] > 0
        points_cam = points_cam[valid_idx]
        points_world_valid = points_world[valid_idx]

        # Proietta nel piano immagine
        proj_2d = K @ points_cam.T
        proj_2d = (proj_2d[:2] / proj_2d[2]).T  # (N, 2)

        # Matcha ogni corner 2D con punto 3D più vicino
        tree = cKDTree(proj_2d)
        matches_3d = []
        for pt2d in corners:
            dist, idx = tree.query(pt2d, distance_upper_bound=max_pixel_error)
            if idx < len(points_world_valid):
                matches_3d.append(points_world_valid[idx])
            else:
                matches_3d.append(None)

        # Filtra match validi
        corners_filtered = []
        matched_3d_filtered = []
        for pt2d, pt3d in zip(corners, matches_3d):
            if pt3d is not None:
                corners_filtered.append(pt2d)
                matched_3d_filtered.append(pt3d)

        results.append((np.array(corners_filtered), np.array(matched_3d_filtered)))

    return results, points_world, pcd


PROCESS_ALL_IMAGES = "Multi-Camera"
class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser


def _convert_scene_output_to_glb(outfile, imgs, pts3d, all_pts3d_object, mask, all_msk_obj, focals, cams2world, cam_size,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, mask_floor=True, mask_objects=False, h2e_list=None, opt_process=None, scale_factor=None, objects=None, intrinsic_params=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    #     all_pts3d_object = array_of_arrays
    if all_pts3d_object is not None:
        max_len = max(len(obj) for obj in all_pts3d_object)
        array_of_arrays = np.empty((max_len, len(all_pts3d_object)), dtype=object)

        for i, obj in enumerate(all_pts3d_object):
            for j in range(max_len):
                if j < len(obj):
                    array_of_arrays[j, i] = obj[j]
                else:
                    array_of_arrays[j, i] = None

        all_pts3d_object = array_of_arrays
    # full pointcloud
    if as_pointcloud:
        
        # Combine all points and their colors
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        valid_pts = pts[valid_msk]
        valid_col = col[valid_msk]
        if all_pts3d_object is not None:
            for i, obj_name in enumerate(objects):
                # if obj_name != "floor":
                #     continue

                pts3d_object = all_pts3d_object[i]
                if pts3d_object is not None:
                    pts3d_object = to_numpy(pts3d_object)
                    
                    valid_pts3d_object = [p for p in pts3d_object if p is not None]

                    # Perform the concatenation only on valid (non-None) points
                    pts_obj = np.concatenate([p[m.ravel()] for p, m in zip(valid_pts3d_object, all_msk_obj[i])]).reshape(-1, 3)
                    # col_obj = np.ones_like(pts_obj) * [1, 0, 0]
                    valid_mask_obj = np.isfinite(pts_obj.sum(axis=1))
                    valid_pts_obj = pts_obj[valid_mask_obj]

                    # Generate a random color (RGB values between 0 and 1)
                    obj_random_color = np.random.rand(3)  # Random color for the object
                    obj_color_mask = np.isin(valid_pts, valid_pts_obj, assume_unique=False).all(axis=1)
                    if mask_objects and obj_name != "floor":
                        # valid_col[obj_color_mask] = [1, 0, 0]  # Red color for object points
                        valid_col[obj_color_mask] = obj_random_color
                    if opt_process == "Mobile-robot" and obj_name == "floor":
                        floor_random_color = np.random.rand(3)  # Random color for the floor
                        # Prepare Open3D point cloud and apply RANSAC for plane fitting
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(valid_pts_obj)

                        # Apply RANSAC for ground plane segmentation
                        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=1000)
                        a, b, c, d = plane_model
                        floor_color_mask = np.isin(valid_pts, valid_pts_obj, assume_unique=False).all(axis=1)
                        if mask_floor:
                            # valid_col[obj_color_mask] = [1, 0, 0]  # Red color for object points
                            valid_col[floor_color_mask] = floor_random_color  # Random color for object points

        pct_updated = trimesh.PointCloud(valid_pts, colors=valid_col)
        scene.add_geometry(pct_updated)

        camera_poses = cams2world  

        camera_frames = []
        heights = []

        if opt_process=="Mobile-robot":
            for i, pose in enumerate(camera_poses):
                if i == 0:
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
                else:
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                camera_frame.transform(pose)
                camera_frames.append(camera_frame)
                camera_position = pose[:3, 3]
                x, y, z = camera_position
                height = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
                heights.append(height)
                # print(f"Height of the camera {i} w.r.t. the ground: {height}")

            if scale_factor is not None:
                height_groups = {}
                for i, height in enumerate(heights):
                    idx = i // int(len(heights)/len(h2e_list))
                    if idx not in height_groups:
                        height_groups[idx] = []

                    height_groups[idx].append(height)
                
                mean_heights = {idx: sum(heights)/len(heights) for idx, heights in height_groups.items()}
                # print(f"Mean height of the cameras w.r.t. the ground: {mean_heights}")
                for i, h2e in enumerate(h2e_list):
                    h2e[2, 3] = mean_heights[i]
                
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)
        
        rob_axis = trimesh.creation.axis(origin_size=0.02, axis_length=0.5)
        robot_centers = []
       
        
        if len(h2e_list) > 0:
            idx = i // int(len(cams2world)/len(h2e_list))
            ee_to_rob = np.array([[9.99807842e-01, 9.69427142e-04, -1.95790426e-02, 5.55120952e-01],
                                [9.77803362e-04, -9.99999434e-01, 4.18246983e-04, 9.85213614e-05],
                                [-1.95786260e-02, -4.37311068e-04, -9.99808225e-01, 5.11640885e-01],
                                [0., 0., 0., 1.]])
            # ee_to_rob = np.array([[0.99925415, 0.03416702, -0.01799314, -0.06077267],
            #                     [0.02872165, -0.96907261, -0.24509866, -0.04276322],
            #                     [-0.02581095, 0.24439906, -0.96933116, 0.47765159],
            #                     [0., 0., 0., 1.]])
            rob_axis.apply_transform(pose_c2w@np.linalg.inv(h2e_list[idx]))
            robot_pose = pose_c2w @ np.linalg.inv(h2e_list[idx])
            robot_centers.append(robot_pose[:3, 3])  
            # scene.add_geometry(rob_axis)
        
        if i == 0:
            cam_axis = trimesh.creation.axis(origin_size=0.02, axis_length=0.2)
            cam_axis.apply_transform(pose_c2w)
            # scene.add_geometry(cam_axis)
            scene.add_geometry(rob_axis)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    print("Camera 0 pose: ", cams2world[0])
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    # Add global frame
    global_frame = trimesh.creation.axis(origin_size=0.01, axis_length=0.5)
    scene.add_geometry(global_frame)

    # if not silent:
    #     print('(exporting 3D scene to', outfile, ')')
    # scene.export(file_obj="pointcloud.glb")
    
    # SAVE PCL FILE
    T_world_to_scene = np.linalg.inv(cams2world[0] @ OPENGL @ rot)
    pct_updated.apply_transform(T_world_to_scene)

    # Ora salva la pcl ruotata
    ply_outfile = os.path.join('pointcloud.ply')
    pct_updated.export(ply_outfile)

    # # METRIC EVALUATION
    # print("Intrinsic parameters: ", intrinsic_params)
    # K=np.array([[intrinsic_params['focal'], 0, intrinsic_params['pp'][0]],
    #             [0, intrinsic_params['focal'], intrinsic_params['pp'][1]],
    #             [0, 0, 1]])
    # # pattern_size = (9, 6)
    # pattern_size = (6, 5)
    # scale_factor = 2.0

    # # Raddoppia le dimensioni di tutte le immagini
    # resized_imgs = []
    # for img in imgs:
    #     h, w = img.shape[:2]
    #     img_uint8 = (img * 255).astype(np.uint8)
    #     resized = cv2.resize(img_uint8, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LINEAR)
    #     resized_imgs.append(resized.astype(np.float32) / 255.0)

    # # Aggiorna K
    # K_scaled = K.copy()
    # K_scaled[0, 0] *= scale_factor  # fx
    # K_scaled[1, 1] *= scale_factor  # fy
    # K_scaled[0, 2] *= scale_factor  # cx
    # K_scaled[1, 2] *= scale_factor  # cy

    # results, _, pcd = extract_3d_chessboard_corners(
    #     imgs=resized_imgs,      
    #     camera_poses=cams2world,
    #     K=K_scaled,
    #     pcd_path="pointcloud.ply",
    #     pattern_size=pattern_size,
    #     max_pixel_error=2
    # )

    # valid_results = []  # salva i risultati validi

    # for i, (pts2d, pts3d) in enumerate(results):
    #     if pts2d is not None:
    #         print(f"[Image {i}] Found {len(pts2d)} corner matches.")
    #     else:
    #         print(f"[Image {i}] No corners detected.")

    # print("\n=== Per-Image Checkerboard Square Size Analysis ===")
    # for i, (pts2d, pts3d) in enumerate(results):
    #     print(f"\n[Image {i}]")
    #     if pts2d is not None and pts3d is not None and len(pts3d) == pattern_size[0] * pattern_size[1]:
    #         horiz, vert = compute_chessboard_square_sizes(pts3d, pattern_size)
    #         mean_h = np.mean(horiz)
    #         mean_v = np.mean(vert)
    #         std_h = np.std(horiz)
    #         std_v = np.std(vert)

    #         print(f"  Mean horizontal square size: {mean_h:.3f} m")
    #         print(f"  Mean vertical square size:   {mean_v:.3f} m")
    #         print(f"  Std horizontal: {std_h:.3f} m, Std vertical: {std_v:.3f} m")
    #         gt = 0.10
    #         # Criteri per accettare la misura come "valida"
    #         if abs(mean_h - gt) < 0.01 and abs(mean_v - gt) < 0.01 and std_h < 0.01 and std_v < 0.01:
    #             valid_results.append((mean_h, mean_v, std_h, std_v))
    #         else:
    #             print("  ⚠️ Risultato scartato: deviazione o media troppo distante da 3 cm")
    #     else:
    #         print("  Not enough 3D corner points for full pattern.")

    # # Report globale
    # if len(valid_results) > 0:
    #     valid_h = [r[0] for r in valid_results]
    #     valid_v = [r[1] for r in valid_results]
    #     all_std_h = [r[2] for r in valid_results]
    #     all_std_v = [r[3] for r in valid_results]

    #     print("\n=== Global Checkerboard Validation (Filtered) ===")
    #     print(f"✅ Images used: {len(valid_results)}")
    #     print(f"✅ Mean horizontal size: {np.mean(valid_h):.4f} m ± {np.std(valid_h):.4f}")
    #     print(f"✅ Mean vertical size:   {np.mean(valid_v):.4f} m ± {np.std(valid_v):.4f}")

    #     from matplotlib import pyplot as plt
    #     plt.errorbar(range(len(valid_h)), valid_h, yerr=all_std_h, fmt='o')
    #     plt.axhline(gt, linestyle='--', color='gray', label=f'Ground Truth ({gt} cm)')
    #     plt.xlabel('Image ID')
    #     plt.ylabel('Mean horizontal square size (m)')
    #     plt.legend()
    #     plt.title('Checkerboard Square Size Estimation per Image')
    #     plt.savefig('checkerboard_square_size.png')
    #     plt.close()
    # else:
    #     print("\n❌ Nessuna immagine ha passato i criteri di validazione.")


    if scale_factor is not None:
        for i in range(len(h2e_list)):
            print(f"Estimated H2E calibration matrix of camera {i+1}: {h2e_list[i]}")

    # Save a txt file with translation and rotation of the cameras only for evaluation
    if scale_factor is not None:
        file_path = "data/pick_objects_1/gt.txt"  # Change this to your actual file path
        gt_transformations = read_transformations(file_path)
        print("GT Transformation: ", gt_transformations)
        est_transformations = []
        # print("GT Transformation: ", gt_transformations)
        for i in range(len(h2e_list)):
            R_h2e = h2e_list[i][:3, :3]
            roll, pitch, yaw = rotation_matrix_to_rpy(R_h2e)
            # print(f'{h2e_list[i][0,3]},{h2e_list[i][1,3]},{h2e_list[i][2,3]}, {roll},{pitch},{yaw}')
            est_transformations.append([h2e_list[i][0,3], h2e_list[i][1,3], h2e_list[i][2,3], roll, pitch, yaw])
        for i in range(len(h2e_list)):
            for j in range(len(h2e_list)):
                if i != j:
                    rel_cam = np.linalg.inv(h2e_list[i]) @ h2e_list[j]
                    R_rel = rel_cam[:3, :3]
                    roll, pitch, yaw = rotation_matrix_to_rpy(R_rel)
                    est_transformations.append([rel_cam[0,3], rel_cam[1,3], rel_cam[2,3], roll, pitch, yaw])
                    # print(f'{rel_cam[0,3]}, {rel_cam[1,3]},{rel_cam[2,3]}, {roll}, {pitch}, {yaw}')

        # print("EST Transformation: ", est_transformations)
        camera_selected = 0
        trans_error, rot_error = compute_errors(np.array(gt_transformations), np.array(est_transformations), len(h2e_list), camera_selected)
        print("Translation error: ", trans_error)
        print("Rotation error: ", rot_error)
        if len(h2e_list) > 1:
            print("Translation error cam2rob: ", np.mean(trans_error[:len(h2e_list)]))
            print("Rotation error cam2rob: ", np.mean(rot_error[:len(h2e_list)]))
        
            print("Translation error cam2cam: ", np.mean(trans_error[len(h2e_list):]))
            print("Rotation error cam2cam: ", np.mean(rot_error[len(h2e_list):]))
        
    return outfile


def get_3D_model_from_scene(silent, scene_state, cam_size, min_conf_thr=2, as_pointcloud=False, mask_sky=False, mask_floor=False, mask_objects=False, calibration_process="Mobile-robot",
                            clean_depth=False, transparent_cams=False, TSDF_thresh=0, objects=None, intrinsic_params=None):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs

    # if scene.corners_2d is not None:
    #     print("Corners 2D: ", scene.corners_2d)

    if scene.masks is not None:
        masks = scene.masks[0]
    else:
        masks = None
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    scale_factor = scene.get_scale_factor()
    quat_X = scene.get_quat_X()
    trans_X = scene.get_trans_X()

    h2e_list = []
    if scale_factor is not None:
        
        for i in range(len(scale_factor)):
            scale_factor[i] = abs(scale_factor[i])
            quat_np = quat_X[i].detach().cpu().numpy()  # Convert PyTorch tensor to NumPy
            norm = np.linalg.norm(quat_np)
            if np.isnan(norm) or norm == 0:
                quat_np = np.array([1, 0, 0, 0])
            rotation = Rotation.from_quat(quat_np, scalar_first=True)  # Create a Rotation object
            h2e = np.eye(4)
            h2e[:3, :3] = rotation.as_matrix()  # Now as_matrix() will work

            h2e[:3, 3] = scale_factor[i].detach().cpu().numpy() * trans_X[i].detach().cpu().numpy()
            h2e_list.append(h2e)

    relative_transformations = []

    for i in range(1, cams2world.shape[0]):
        T_prev_inv = torch.inverse(cams2world[i - 1])   # Inverse of previous transformation
        T_current = cams2world[i]                       # Current transformation
        T_rel = T_prev_inv @ T_current                  # Relative transformation
        relative_transformations.append(T_rel)

    # Convert list to a tensor for easier manipulation if desired
    relative_transformations = torch.stack(relative_transformations)
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, pts3d_object, _, confs = to_numpy(tsdf.get_dense_pts3d(masks=masks, clean_depth=clean_depth))
    else:
        pts3d, pts3d_object, _, confs, confs_object = to_numpy(scene.get_dense_pts3d(masks=masks, corners=scene.corners_2d, clean_depth=clean_depth))

    msk = to_numpy([c > min_conf_thr for c in confs])
    
    all_msk_obj = []
    if confs_object is not None:
        for conf_object in confs_object:
            valid_conf_object = [c for c in conf_object if c is not None]

            # Now apply the comparison to the valid conf_object elements
            msk_obj = to_numpy([c > min_conf_thr for c in valid_conf_object])
            all_msk_obj.append(msk_obj)
    ccam2pcam = scene.get_relative_poses()
    # ccam2pcam = scene.get_im_poses().cpu()
    print("Scale factor: ", scale_factor)
    if scale_factor is not None:
        cams2world = ccam2pcam
    else:
        cams2world = cams2world

    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, pts3d_object, msk, all_msk_obj, focals, cams2world, cam_size, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, silent=silent, mask_floor=mask_floor, mask_objects=mask_objects, h2e_list=h2e_list, opt_process=calibration_process, scale_factor=scale_factor, objects=objects, intrinsic_params=intrinsic_params)



def detect_checkerboard_and_flatten(filelist, pattern_size=(9, 6), target_shape=(288, 512)):
    checkerboard_corners_resized = []

    print("Target shape: ", target_shape)
    for path in filelist:
        # Load original image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Failed to load image at {path}")
            continue

        original_h, original_w = img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect checkerboard corners in original image
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refine corner positions in original image
            corners_subpix = cv2.cornerSubPix(
                gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            
            # Scale corners to match resized image
            scale_x = target_shape[1] / original_w
            scale_y = target_shape[0] / original_h
            corners_resized = corners_subpix.squeeze(1) * np.array([scale_x, scale_y])
            checkerboard_corners_resized.append(corners_resized)  # (N, 2)
        else:
            checkerboard_corners_resized.append(None)
            print(f"[WARN] Checkerboard not found in {path}")

    return checkerboard_corners_resized


def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, config, 
                            camera_to_use, flattened_filelist, camera_num, intrinsic_params, dist_coeffs, robot_poses, calibration_process, multiple_camera_opt, lr1, niter1, as_pointcloud, mask_sky, 
                            mask_floor, mask_objects, clean_depth, transparent_cams, scenegraph_type, winsize,
                            win_cyclic, input_text_prompt, metric_evaluation, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    opt_process = PROCESS_ALL_IMAGES if camera_to_use == 0 else f"Camera {camera_to_use}"
    flattened_imgs, _ = load_single_images(flattened_filelist, config['image_size'], verbose=not config['silent'])
    
    image_ext = None
    objects = None
    print("FLATTENED FILELIST: ", flattened_filelist)
    print("Flattened imgs shape: ", flattened_imgs[0]['img'].shape)

    corners_2d = None
    if metric_evaluation:
        corners_2d = detect_checkerboard_and_flatten(flattened_filelist, pattern_size=(6, 5), target_shape=flattened_imgs[0]['img'].shape[2:])

    mask_generator = MaskGenerator(config, flattened_filelist, input_text_prompt, calibration_process)
    if mask_generator.start_mask() == True:
        print("Generating masks...")
        objects, image_ext = mask_generator.generate_masks()
        subfolders = mask_generator.get_subfolders()
        image_sublist = mask_generator.get_image_list()

        mask_list = generate_mask_list(subfolders, image_sublist, image_ext=image_ext)
    else:
        mask_list = None

    if mask_list is not None and len(mask_list) > 0:
        if camera_to_use == 0:
            flattened_masklist = [item for sublist in mask_list for item in sublist]
        else:
            flattened_masklist = mask_list
        flattened_msks = load_single_masks(flattened_masklist, flattened_filelist, size=config['image_size'], verbose=not config['silent'])
    else:
        flattened_msks = None

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    pairs = make_pairs(flattened_imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')

    os.makedirs(cache_dir, exist_ok=True)

    if isinstance(intrinsic_params, list) and all(item is None for item in intrinsic_params):
        intrinsic_params = None
    print("> Start global optimization")
    scene = sparse_global_alignment(flattened_filelist, pairs, cache_dir,
                                    model, opt_process, camera_num, flattened_msks, corners_2d, intrinsic_params=intrinsic_params, dist_coeffs_cam=dist_coeffs, 
                                    robot_poses=robot_poses, multiple_camera_opt=multiple_camera_opt, lr1=lr1, niter1=niter1, device=device,
                                    opt_depth=True, shared_intrinsics=config['shared_intrinsics'],
                                    matching_conf_thr=config['matching_conf_thr'], **kw)

    outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, config['cam_size'], config['min_conf_thr'], as_pointcloud, mask_sky, mask_floor, mask_objects, calibration_process,
                                      clean_depth, transparent_cams, config['TSDF_thresh'], objects, intrinsic_params)
    
    images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in flattened_filelist]
    return scene_state, outfile, images


def set_scenegraph_options(inputfiles, win_cyclic, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    show_win_controls = scenegraph_type in ["swin", "logwin"]
    show_winsize = scenegraph_type in ["swin", "logwin"]
    show_cyclic = scenegraph_type in ["swin", "logwin"]
    max_winsize, min_winsize = 1, 1
    if scenegraph_type == "swin":
        if win_cyclic:
            max_winsize = max(1, math.ceil((num_files - 1) / 2))
        else:
            max_winsize = num_files - 1
    elif scenegraph_type == "logwin":
        if win_cyclic:
            half_size = math.ceil((num_files - 1) / 2)
            max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        else:
            max_winsize = max(1, math.ceil(math.log(num_files, 2)))
    winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                            minimum=min_winsize, maximum=max_winsize, step=1, visible=show_winsize)
    win_cyclic = gradio.Checkbox(value=win_cyclic, label="Cyclic sequence", visible=show_cyclic)
    win_col = gradio.Column(visible=show_win_controls)

    return win_col, winsize, win_cyclic


def main_demo(tmpdirname, model, config, device, input_images=None, silent=False, camera_num=None, intrinsic_params=None, dist_coeffs=None, robot_poses=None, mask_floor=False, camera_to_use=0, calibration_process="Robot-Arm", 
              multiple_camera_opt=True, input_text_prompt="", metric_evaluation=False, share=False, gradio_delete_cache=False):

    if not silent:
        print('Outputting stuff in', tmpdirname)

    get_reconstructed_scene(tmpdirname, gradio_delete_cache, model, device, silent, config, 
                        camera_to_use, input_images, camera_num, intrinsic_params, dist_coeffs, robot_poses, calibration_process, multiple_camera_opt, lr1=0.07, niter1=500, as_pointcloud=True, mask_sky=False, 
                        mask_floor=mask_floor, mask_objects=False, clean_depth=True, transparent_cams=False, scenegraph_type="complete", winsize=1,
                        win_cyclic=False, input_text_prompt=input_text_prompt, metric_evaluation=metric_evaluation)