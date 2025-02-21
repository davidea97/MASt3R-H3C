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

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, load_masks, load_single_masks, load_images_intr, load_single_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl


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
                                 transparent_cams=False, silent=False, mask_floor=True):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    array_of_arrays = np.array([[None for _ in range(len(all_pts3d_object))] for _ in range(1)], dtype=object)
    for i, pts3d_object in enumerate(all_pts3d_object):
        array = []
        for j in range(1):
            if j < len(pts3d_object):
                array_of_arrays[j][i] = pts3d_object[j]
            else:
                array_of_arrays[j][i] = None
            
    all_pts3d_object = array_of_arrays


    # full pointcloud
    if as_pointcloud:
        
        # Combine all points and their colors
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        valid_pts = pts[valid_msk]
        valid_col = col[valid_msk]

        for i, pts3d_object in enumerate(all_pts3d_object):
            if pts3d_object is not None:
                pts3d_object = to_numpy(pts3d_object)
                
                valid_pts3d_object = [p for p in pts3d_object if p is not None]

                # Perform the concatenation only on valid (non-None) points
                pts_obj = np.concatenate([p[m.ravel()] for p, m in zip(valid_pts3d_object, all_msk_obj[i])]).reshape(-1, 3)
                # col_obj = np.ones_like(pts_obj) * [1, 0, 0]
                valid_mask_obj = np.isfinite(pts_obj.sum(axis=1))
                valid_pts_obj = pts_obj[valid_mask_obj]

                # Generate a random color (RGB values between 0 and 1)
                random_color = np.random.rand(3)  # Random color for the object
                reference_frame_object = None
                if mask_floor:
                    
                    # Prepare Open3D point cloud and apply RANSAC for plane fitting
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(valid_pts_obj)


                    # centroid = np.mean(np.asarray(pcd.points), axis=0)

                    # reference_frame_object = trimesh.creation.axis(origin_size=0.05, axis_length=0.4)
                    # reference_frame_object.apply_translation(centroid)
                    # # Rotation matrix for 180 degrees around the Y-axis
                    # rotation_matrix_x = trimesh.transformations.rotation_matrix(
                    #     angle=np.pi/2+0.3,  # 180 degrees in radians
                    #     direction=[1, 0, 0],  # Y-axis
                    #     point=centroid
                    # )

                    # # Apply the rotation to the reference frame
                    # reference_frame_object.apply_transform(rotation_matrix_x)

                    # rotation_matrix_z = trimesh.transformations.rotation_matrix(
                    #     angle=np.pi/4,  # 180 degrees in radians
                    #     direction=[0, 0, 1],  # Y-axis
                    #     point=centroid
                    # )
                    # Visualize the segmented plane with the reference frame
                    # o3d.visualization.draw_geometries([pcd, plane_cloud, reference_frame])

                    # reference_frame_object.apply_transform(rotation_matrix_z)
                    # Apply RANSAC for ground plane segmentation
                    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=1000)
                    a, b, c, d = plane_model
                    obj_color_mask = np.isin(valid_pts, valid_pts_obj, assume_unique=False).all(axis=1)

                    # valid_col[obj_color_mask] = [1, 0, 0]  # Red color for object points
                    valid_col[obj_color_mask] = random_color  # Random color for object points

                pct_updated = trimesh.PointCloud(valid_pts, colors=valid_col)
                scene.add_geometry(pct_updated)
                # scene.add_geometry(reference_frame_object)
        
            # Update point cloud with new colors
            # pct_updated = trimesh.PointCloud(valid_pts, colors=valid_col)
            # scene.add_geometry(pct_updated)
            
            camera_poses = cams2world  
            camera_frames = []
            heights = []
            reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            pose_camera_0 = np.eye(4)
            
            for i, pose in enumerate(camera_poses):
                if i == 0:
                    pose_camera_0 = pose
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
                else:
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                camera_frame.transform(pose)  
                camera_frames.append(camera_frame)
                camera_position = pose[:3, 3]
                x, y, z = camera_position

                if mask_floor:
                    height = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
                    heights.append(height)
                    print(f"Height of the camera {i} w.r.t. the ground: {height}")
        else: 
            # Prepare Open3D point cloud and apply RANSAC for plane fitting
            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(pts)
            # Apply RANSAC for ground plane segmentation
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=1000)
            a, b, c, d = plane_model

            camera_poses = cams2world  
            heights = []
            
            for i, pose in enumerate(camera_poses):
                if i == 0:
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
                else:
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                camera_frame.transform(pose)
                # camera_frames.append(camera_frame)
                camera_position = pose[:3, 3]
                x, y, z = camera_position

                # Calcola l'altezza rispetto al pavimento
                height = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
                heights.append(height)
                print(f"Height of the camera {i} w.r.t. the ground: {height}")
        
        # colors = col[valid_msk]
        # pct = trimesh.PointCloud(pts, colors=colors)
        # scene.add_geometry(pct)

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
        
        if i == 0:
            cam_axis = trimesh.creation.axis(origin_size=0.02, axis_length=0.2)
            cam_axis.apply_transform(pose_c2w)
            scene.add_geometry(cam_axis)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    # Add global frame
    global_frame = trimesh.creation.axis(origin_size=0.01, axis_length=0.1)
    # scene.add_geometry(global_frame)

    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(silent, scene_state, cam_size, min_conf_thr=2, as_pointcloud=False, mask_sky=False, mask_floor=False, mask_objects=False,
                            clean_depth=False, transparent_cams=False, TSDF_thresh=0):
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
    masks = scene.masks[0]
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    scale_factor = scene.get_scale_factor()
    quat_X = scene.get_quat_X()
    trans_X = scene.get_trans_X()

    ccam2pcam = scene.get_relative_poses()
    
    if scale_factor is not None:
        print(f"Estimated scale factor: {scale_factor}")
        print(f"Estimated quaternion X: {quat_X}")
        print(f"Estimated scaled translation X: {scale_factor*trans_X}")

        quat_np = quat_X.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy
        rotation = Rotation.from_quat(quat_np)  # Create a Rotation object
        h2e = np.eye(4)
        h2e[:3, :3] = rotation.as_matrix()  # Now as_matrix() will work

        h2e[:3, 3] = scale_factor.detach().cpu().numpy() * trans_X.detach().cpu().numpy()
        print(f"Estimated H2E calibration matrix: {h2e}")

    relative_transformations = []

    for i in range(1, cams2world.shape[0]):
        T_prev_inv = torch.inverse(cams2world[i - 1])   # Inverse of previous transformation
        T_current = cams2world[i]                       # Current transformation
        T_rel = T_prev_inv @ T_current                  # Relative transformation
        relative_transformations.append(T_rel)

    # Save each relative transformation as a YAML file in OpenCV format
    # for i, T_rel in enumerate(relative_transformations):
    #     # Convert PyTorch tensor to NumPy array
    #     T_rel_np = T_rel.numpy().astype(np.float64)  # OpenCV YAML format expects double precision (float64)

    #     # Define the output file name
    #     filename = f"relative_cam_pose_{i}.yaml"

    #     # Open a cv2.FileStorage for writing in YAML format
    #     fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    #     fs.write("matrix", T_rel_np)
    #     fs.release()

    #     print(f"Saved relative transformation to {filename}")

    # Convert list to a tensor for easier manipulation if desired
    relative_transformations = torch.stack(relative_transformations)

    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, pts3d_object, _, confs = to_numpy(tsdf.get_dense_pts3d(masks=masks, clean_depth=clean_depth))
    else:
        pts3d, pts3d_object, _, confs, confs_object = to_numpy(scene.get_dense_pts3d(masks=masks, clean_depth=clean_depth))

    msk = to_numpy([c > min_conf_thr for c in confs])
    
    all_msk_obj = []
    for conf_object in confs_object:
        valid_conf_object = [c for c in conf_object if c is not None]

        # Now apply the comparison to the valid conf_object elements
        msk_obj = to_numpy([c > min_conf_thr for c in valid_conf_object])
        all_msk_obj.append(msk_obj)

    if scale_factor is not None:
        cams2world = ccam2pcam
    else:
        cams2world = cams2world

    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, pts3d_object, msk, all_msk_obj, focals, cams2world, cam_size, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, silent=silent, mask_floor=mask_floor)



def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, config, 
                            current_scene_state, filelist, intrinsic_params, dist_coeffs, mask_list, robot_poses, optim_level, lr1, niter1, as_pointcloud, mask_sky, 
                            mask_floor, mask_objects, clean_depth, transparent_cams, scenegraph_type, winsize,
                            win_cyclic, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    # imgs = load_images(filelist, size=image_size, verbose=not silent)
    msks = None
    # cam_size = config['cam_size']
    imgs, _ = load_single_images(filelist, config['image_size'], verbose=not config['silent'])

    if mask_list is not None and len(mask_list) > 0:
        msks = load_single_masks(mask_list, filelist, size=config['image_size'], verbose=not config['silent'])

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, msks, intrinsic_params=intrinsic_params, dist_coeffs_cam=dist_coeffs, 
                                    robot_poses=robot_poses, lr1=lr1, niter1=niter1, device=device,
                                    opt_depth=optim_level, shared_intrinsics=config['shared_intrinsics'],
                                    matching_conf_thr=config['matching_conf_thr'], **kw)

    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, config['cam_size'], config['min_conf_thr'], as_pointcloud, mask_sky, mask_floor, mask_objects, 
                                      clean_depth, transparent_cams, config['TSDF_thresh'])
    return scene_state, outfile


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


def main_demo(tmpdirname, model, config, device, server_name, server_port, image_list=None, mask_list=None, silent=False, intrinsic_params_vec=None, dist_coeffs=None, robot_poses=None,
              share=False, gradio_delete_cache=False):
    
    if not silent:
        print('Outputting stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, gradio_delete_cache, model, device,
                                   silent, config)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R-H3C Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
        else:
            return gradio.Blocks(css=css, title="MASt3R-HEC Demo")  # for compatibility with older versions

    def process_images(scene, input_images, intrinsic_params, dist_coeff, masks, robot_pose, optim_level, lr1, niter1,
                       as_pointcloud, mask_sky, mask_floor, mask_objects, clean_depth, transparent_cams, scenegraph_type,
                       winsize, win_cyclic):
        if isinstance(input_images, list) and all(isinstance(img, str) for img in input_images):
            # Use provided image list
            return recon_fun(scene, input_images, intrinsic_params, dist_coeff, masks, robot_pose, optim_level,
                             lr1, niter1, as_pointcloud, mask_sky, mask_floor, mask_objects, clean_depth, transparent_cams, scenegraph_type,
                             winsize, win_cyclic)

        else:
            raise ValueError("No valid input images provided!")

    with get_context(gradio_delete_cache) as demo:
        # Scene state is saved so that you can change parameters without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">MASt3R-HEC Demo</h2>')

        with gradio.Column():
            if image_list:
                gradio.HTML('<h3>Provided Images:</h3>')
                
                # Dropdown to select the image list
                list_selector = gradio.Dropdown(
                    choices=[f"List {i+1}" for i in range(len(image_list))],
                    value="Select the image list to process",
                    label="Select Image List",
                )
                
                def update_image_list(selected_list):
                    # Extract the list index from the selection
                    index = int(selected_list.split()[1]) - 1
                    print("Selected image list index: ", index)
                    print("Image list : ", image_list)
                    single_image_list = image_list[index]
                    intrinsic_params = intrinsic_params_vec[index] if intrinsic_params_vec else None
                    dist_coeff = dist_coeffs[index] if dist_coeffs else None
                    robot_pose = robot_poses[index] if robot_poses else None
                    masks = mask_list[index] if mask_list else None


                    # Format the list as a newline-separated string
                    formatted_list = "\n".join(single_image_list)
                    return formatted_list, single_image_list, intrinsic_params, dist_coeff, robot_pose, masks
                
                # Dynamically display the selected image paths
                selected_image_paths = gradio.Textbox(
                    label="Selected Image Paths",
                    lines=5,  # Adjust to show more lines
                    interactive=False
                )

                # Placeholder for dynamically updated file input
                inputfiles = gradio.File(
                    file_count="multiple",
                    label="Files to Process"
                )

                intrinsic_state = gradio.State(None)
                dist_coeff_state = gradio.State(None)
                robot_pose_state = gradio.State(None)
                masks_state = gradio.State(None)


                # Update handlers for list selection
                list_selector.change(
                    fn=update_image_list,
                    inputs=list_selector,
                    outputs=[selected_image_paths, inputfiles, intrinsic_state, dist_coeff_state, robot_pose_state, masks_state]
                )
            else:
                # Fall back to file upload if no image_list is provided
                gradio.HTML('<h3>Upload Images:</h3>')
                inputfiles = gradio.File(file_count="multiple", label="Upload Images")

            # Configurable settings
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
                        niter1 = gradio.Number(value=500, precision=0, minimum=0, maximum=10_000,
                                               label="num_iterations", info="For coarse alignment!")
                        
                        optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
                                                      value='refine+depth', label="OptLevel",
                                                      info="Optimization level")
                        
                    with gradio.Row():
                        scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
                                                           ("swin: sliding window", "swin"),
                                                           ("logwin: sliding window with long range", "logwin"),
                                                           ("oneref: match one image with all", "oneref")],
                                                          value='complete', label="Scenegraph",
                                                          info="Define how to make pairs",
                                                          interactive=True)
                        with gradio.Column(visible=False) as win_col:
                            winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                                    minimum=1, maximum=1, step=1)
                            win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
            run_btn = gradio.Button("Run 3D reconstruction")

            # with gradio.Row():
            #     cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                mask_floor = gradio.Checkbox(value=False, label="Mask floor")
                mask_objects = gradio.Checkbox(value=False, label="Mask objects")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, scenegraph_type],
                                   outputs=[win_col, winsize, win_cyclic])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic])

            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic])

            

            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, as_pointcloud, mask_sky, mask_floor, mask_objects,
                                         clean_depth, transparent_cams],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, as_pointcloud, mask_sky,mask_floor, mask_objects,
                                    clean_depth, transparent_cams],
                            outputs=outmodel)
            mask_floor.change(fn=model_from_scene_fun,
                               inputs=[scene, as_pointcloud, mask_sky, mask_floor, mask_objects,
                                       clean_depth, transparent_cams],
                               outputs=outmodel)
            mask_objects.change(fn=model_from_scene_fun,
                               inputs=[scene, as_pointcloud, mask_sky, mask_floor, mask_objects,
                                       clean_depth, transparent_cams],
                               outputs=outmodel)

            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, as_pointcloud, mask_sky,mask_floor, mask_objects,
                                       clean_depth, transparent_cams],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, as_pointcloud, mask_sky,mask_floor, mask_objects,
                                            clean_depth, transparent_cams],
                                    outputs=outmodel)

            run_btn.click(
                fn=process_images,
                inputs=[scene, inputfiles, intrinsic_state, dist_coeff_state, masks_state, robot_pose_state, optim_level, lr1, niter1, as_pointcloud, mask_sky, mask_floor, mask_objects,
                        clean_depth, transparent_cams, scenegraph_type, winsize,
                        win_cyclic],
                outputs=[scene, outmodel]
            )

    demo.launch(share=share, server_name=server_name, server_port=server_port)
