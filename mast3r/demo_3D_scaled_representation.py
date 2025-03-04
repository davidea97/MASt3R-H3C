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
from dust3r.utils.image import load_single_masks, load_single_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser
from mast3r.utils.general_utils import reshape_list
import matplotlib.pyplot as pl

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
                                 transparent_cams=False, silent=False, mask_floor=True, h2e_list=None, opt_process=None, scale_factor=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    if all_pts3d_object is not None:
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
        if all_pts3d_object is not None:
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
                    if opt_process == "Mobile-robot":
                        
                        # Prepare Open3D point cloud and apply RANSAC for plane fitting
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(valid_pts_obj)

                        # Apply RANSAC for ground plane segmentation
                        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=1000)
                        a, b, c, d = plane_model
                        obj_color_mask = np.isin(valid_pts, valid_pts_obj, assume_unique=False).all(axis=1)
                        if mask_floor:
                            # valid_col[obj_color_mask] = [1, 0, 0]  # Red color for object points
                            valid_col[obj_color_mask] = random_color  # Random color for object points

        pct_updated = trimesh.PointCloud(valid_pts, colors=valid_col)
        scene.add_geometry(pct_updated)
        
        camera_poses = cams2world  
        camera_frames = []
        heights = []

        for i, pose in enumerate(camera_poses):
            if i == 0:
                camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            else:
                camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            camera_frame.transform(pose)  
            camera_frames.append(camera_frame)
            camera_position = pose[:3, 3]
            x, y, z = camera_position

            if opt_process=="Mobile-robot":
                height = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
                heights.append(height)
                # print(f"Height of the camera {i} w.r.t. the ground: {height}")

        if opt_process=="Mobile-robot" and scale_factor is not None:
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
    if scale_factor is not None:
        for i in range(len(h2e_list)):
            print(f"Estimated H2E calibration matrix of camera {i+1}: {h2e_list[i]}")
    return outfile


def get_3D_model_from_scene(silent, scene_state, cam_size, min_conf_thr=2, as_pointcloud=False, mask_sky=False, mask_floor=False, calibration_process="Mobile-robot",
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

    h2e_list = []
    if scale_factor is not None:
        
        # print(f"Estimated scale factor: {scale_factor}")
        # print(f"Estimated quaternion X: {quat_X}")
        # print(f"Estimated translation X: {trans_X}")
        # print(f"Estimated scaled translation X: {[scale_factor[i]*trans_X[i] for i in range(len(scale_factor))]}")
        for i in range(len(scale_factor)):
            scale_factor[i] = abs(scale_factor[i])
            quat_np = quat_X[i].detach().cpu().numpy()  # Convert PyTorch tensor to NumPy
            rotation = Rotation.from_quat(quat_np)  # Create a Rotation object
            h2e = np.eye(4)
            h2e[:3, :3] = rotation.as_matrix()  # Now as_matrix() will work

            h2e[:3, 3] = scale_factor[i].detach().cpu().numpy() * trans_X[i].detach().cpu().numpy()
            h2e_list.append(h2e)
            # print(f"Estimated H2E calibration matrix of camera {i+1}: {h2e}")

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
        pts3d, pts3d_object, _, confs, confs_object = to_numpy(scene.get_dense_pts3d(masks=masks, clean_depth=clean_depth))

    msk = to_numpy([c > min_conf_thr for c in confs])
    
    all_msk_obj = []
    if confs_object is not None:
        for conf_object in confs_object:
            valid_conf_object = [c for c in conf_object if c is not None]

            # Now apply the comparison to the valid conf_object elements
            msk_obj = to_numpy([c > min_conf_thr for c in valid_conf_object])
            all_msk_obj.append(msk_obj)

    ccam2pcam = scene.get_relative_poses()
    if scale_factor is not None:
        cams2world = ccam2pcam
    else:
        cams2world = cams2world

    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, pts3d_object, msk, all_msk_obj, focals, cams2world, cam_size, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, silent=silent, mask_floor=mask_floor, h2e_list=h2e_list, opt_process=calibration_process, scale_factor=scale_factor)



def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, config, 
                            current_scene_state, flattened_filelist, opt_process, camera_num, intrinsic_params, dist_coeffs, mask_list, robot_poses, calibration_process, lr1, niter1, as_pointcloud, mask_sky, 
                            mask_floor, clean_depth, transparent_cams, scenegraph_type, winsize,
                            win_cyclic, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    
    # folder_list = reshape_list(flattened_filelist, len(robot_poses))

    msks = None
    flattened_imgs, _ = load_single_images(flattened_filelist, config['image_size'], verbose=not config['silent'])
    
    if mask_list is not None and len(mask_list) > 0:
        if opt_process == PROCESS_ALL_IMAGES:
            flattened_masklist = [item for sublist in mask_list for item in sublist]
        else:
            flattened_masklist = mask_list
        flattened_msks = load_single_masks(flattened_masklist, flattened_filelist, size=config['image_size'], verbose=not config['silent'])

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    pairs = make_pairs(flattened_imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

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

    if isinstance(intrinsic_params, list) and all(item is None for item in intrinsic_params):
        intrinsic_params = None

    scene = sparse_global_alignment(flattened_filelist, pairs, cache_dir,
                                    model, opt_process, camera_num, flattened_msks, intrinsic_params=intrinsic_params, dist_coeffs_cam=dist_coeffs, 
                                    robot_poses=robot_poses, lr1=lr1, niter1=niter1, device=device,
                                    opt_depth=True, shared_intrinsics=config['shared_intrinsics'],
                                    matching_conf_thr=config['matching_conf_thr'], **kw)

    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, config['cam_size'], config['min_conf_thr'], as_pointcloud, mask_sky, mask_floor, calibration_process,
                                      clean_depth, transparent_cams, config['TSDF_thresh'])
    
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


def main_demo(tmpdirname, model, config, device, server_name, server_port, image_list=None, mask_list=None, silent=False, camera_num=None, intrinsic_params_vec=None, dist_coeffs=None, robot_poses=None,
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

    def process_images(scene, input_images, opt_process, camera_num, intrinsic_params, dist_coeff, masks, robot_pose, calibration_process, lr1, niter1,
                   as_pointcloud, mask_sky, mask_floor, clean_depth, transparent_cams, scenegraph_type,
                   winsize, win_cyclic):
        if isinstance(input_images, list) and all(isinstance(img, str) for img in input_images):
            # Single list of images
            return recon_fun(scene, input_images, opt_process, camera_num, intrinsic_params, dist_coeff, masks, robot_pose, calibration_process, 
                            lr1, niter1, as_pointcloud, mask_sky, mask_floor, clean_depth, transparent_cams, scenegraph_type,
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
                if len(image_list) > 1:
                    choices = [PROCESS_ALL_IMAGES] + [f"Camera {i+1}" for i in range(len(image_list))]
                else:
                    choices = [f"Camera {i+1}" for i in range(len(image_list))]
                # Dropdown to select the image list
                list_selector = gradio.Dropdown(
                    choices=choices,
                    value="Select the image list to process",
                    label="Select Image List",
                )

                
                def update_image_list(selected_list):
                    if selected_list == PROCESS_ALL_IMAGES:
                        # Use the whole vector of image lists without flattening
                        selected_images = image_list
                        intrinsic_params = intrinsic_params_vec if intrinsic_params_vec else None
                        dist_coeff = dist_coeffs if dist_coeffs else None
                        robot_pose = robot_poses if robot_poses else None
                        masks = mask_list if mask_list else None
                        formatted_list = "\n\n".join(
                            [f"Camera {i+1}:\n" + "\n".join(img_list) for i, img_list in enumerate(selected_images)]
                        )
                        camera_num = len(selected_images)
                        selected_images_flat = [img for sublist in selected_images for img in sublist]

                    else:
                        # Extract the list index from the selection
                        index = int(selected_list.split()[1]) - 1
                        selected_images = image_list[index]
                        intrinsic_params = intrinsic_params_vec[index] if intrinsic_params_vec else None
                        dist_coeff = [dist_coeffs[index]] if dist_coeffs else None
                        # robot_pose = robot_poses[index] if robot_poses else None
                        robot_pose = robot_poses if robot_poses else None
                        masks = mask_list[index] if mask_list else None
                        formatted_list = "\n".join(selected_images) if isinstance(selected_images, list) else "Processing all image lists."
                        selected_images_flat = selected_images
                        camera_num = 1
                    return formatted_list, selected_images_flat, intrinsic_params, dist_coeff, robot_pose, masks, selected_list, camera_num

                
                # Dynamically display the selected image paths
                selected_image_paths = gradio.Textbox(
                    label="Selected Image Paths",
                    lines=5,  # Adjust to show more lines
                    interactive=False
                )

                intrinsic_state = gradio.State(None)
                dist_coeff_state = gradio.State(None)
                robot_pose_state = gradio.State(None)
                masks_state = gradio.State(None)
                opt_process_state = gradio.State(None)
                inputfiles = gradio.State(None)
                camera_num = gradio.State(None)

                # Update handlers for list selection
                list_selector.change(
                    fn=update_image_list,
                    inputs=list_selector,
                    outputs=[selected_image_paths, inputfiles, intrinsic_state, dist_coeff_state, robot_pose_state, masks_state, opt_process_state, camera_num]
                )
            else:
                # Placeholder for dynamically updated file input
                inputfiles = gradio.File(
                    file_count="multiple",
                    label="Files to Process"
                )
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
                        
                        if robot_poses is not None:
                            calibration_process = gradio.Dropdown(["Mobile-robot", "Robot Arm"],
                                                          value='Mobile-robot', label="optimization",
                                                          info="Optimization process",
                                                          interactive=True)
                        
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
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                mask_floor = gradio.Checkbox(value=False, label="Mask floor")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb', columns=4, height="100%")

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
                                 inputs=[scene, as_pointcloud, mask_sky, mask_floor,
                                         clean_depth, transparent_cams],
                                 outputs=outmodel)
            
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, as_pointcloud, mask_sky,mask_floor,
                                    clean_depth, transparent_cams],
                            outputs=outmodel)
            
            mask_floor.change(fn=model_from_scene_fun,
                               inputs=[scene, as_pointcloud, mask_sky, mask_floor,
                                       clean_depth, transparent_cams],
                               outputs=outmodel)

            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, as_pointcloud, mask_sky,mask_floor,
                                       clean_depth, transparent_cams],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, as_pointcloud, mask_sky,mask_floor,
                                            clean_depth, transparent_cams],
                                    outputs=outmodel)

            run_btn.click(
                fn=process_images,
                inputs=[scene, inputfiles, opt_process_state, camera_num, intrinsic_state, dist_coeff_state, masks_state, robot_pose_state, calibration_process, lr1, niter1, as_pointcloud, mask_sky, mask_floor,
                        clean_depth, transparent_cams, scenegraph_type, winsize, win_cyclic],
                outputs=[scene, outmodel, outgallery]
            )

    demo.launch(share=share, server_name=server_name, server_port=server_port)
