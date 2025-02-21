from dust3r.utils.image import load_images, load_masks, load_images_intr
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from scipy.spatial.transform import Rotation

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

import tempfile
import os
import numpy as np
import trimesh
import open3d as o3d
import torch
import cv2
import yaml
import shutil


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


def _convert_scene_output_to_glb(input_dir, camera_num, outfile, imgs, pts3d, pts3d_object, mask, mask_object, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, mask_floor=True):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)
    scene = trimesh.Scene()

    if pts3d_object is not None:
        pts3d_object = to_numpy(pts3d_object)

    # full pointcloud
    if as_pointcloud:
        
        if pts3d_object is not None:
            pts3d_object = [arr for arr in pts3d_object if arr.size > 0]  # Filter out empty arrays
        
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        valid_pts = pts[valid_msk]
        valid_col = col[valid_msk]

        # Ground plane detection (check if the masks are available or not)
        if pts3d_object is not None:
            pts_obj = np.concatenate([p[m.ravel()] for p, m in zip(pts3d_object, mask_object)]).reshape(-1, 3)
            col_obj = np.ones_like(pts_obj) * [1, 0, 0]
            valid_mask_obj = np.isfinite(pts_obj.sum(axis=1))
            valid_pts_obj = pts_obj[valid_mask_obj]
            valid_col_obj = col_obj[valid_mask_obj]

            # Prepare Open3D point cloud and apply RANSAC for plane fitting
            pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.points = o3d.utility.Vector3dVector(valid_pts_obj)
            # Apply RANSAC for ground plane segmentation
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=1000)
            a, b, c, d = plane_model
            # plane_normal = np.array([a, b, c])

            # projection = -d / (a**2 + b**2 + c**2) * plane_normal
            if mask_floor:
                pct = trimesh.PointCloud(valid_pts_obj, colors=valid_col_obj)
                scene.add_geometry(pct)
            
            pct_updated = trimesh.PointCloud(valid_pts, colors=valid_col)
            scene.add_geometry(pct_updated)

            camera_poses = cams2world  

            camera_frames = []
            heights = []
            reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            pose_camera_0 = np.eye(4)
            output_height_dir = os.path.join(input_dir, os.path.join(f"camera{camera_num}", "estimated_camera_heights"))
            if os.path.exists(output_height_dir):
                shutil.rmtree(output_height_dir)
            os.makedirs(output_height_dir, exist_ok=True)
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
                    
                    # Define the output file name
                    filename = os.path.join(output_height_dir, f"{i:04}.yaml")

                    # Open a cv2.FileStorage for writing in YAML format
                    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
                    fs.write("height", height)  # Scrive l'altezza al posto di "matrix"
                    fs.release()
            
            # colors_obj = col_obj[valid_mask_obj]
            # pct_obj = trimesh.PointCloud(pts_obj, colors=colors_obj)
            # scene.add_geometry(pct_obj)
        else: 
            # Prepare Open3D point cloud and apply RANSAC for plane fitting
            pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.points = o3d.utility.Vector3dVector(pts)
            # Apply RANSAC for ground plane segmentation
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=1000)
            a, b, c, d = plane_model
            plane_normal = np.array([a, b, c])

            projection = -d / (a**2 + b**2 + c**2) * plane_normal

            # Aggiungi le pose della camera come frame di coordinate
            camera_poses = cams2world  # Inserisci qui le pose della camera come matrici 4x4
            # camera_poses = ccam2wcam
            
            camera_frames = []
            heights = []
            reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            pose_camera_0 = np.eye(4)
            output_height_dir = os.path.join(input_dir, "estimated_camera_heights")
            if os.path.exists(output_height_dir):
                shutil.rmtree(output_height_dir)
            os.makedirs(output_height_dir, exist_ok=True)
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

                # Calcola l'altezza rispetto al pavimento
                height = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
                heights.append(height)

                print(f"Height of the camera {i} w.r.t. the ground: {height}")
                
                # Define the output file name
                filename = os.path.join(output_height_dir, f"{i:04}.yaml")

                # Open a cv2.FileStorage for writing in YAML format
                fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
                fs.write("height", height)  # Scrive l'altezza al posto di "matrix"
                fs.release()
        
        colors = col[valid_msk]
        pct = trimesh.PointCloud(pts, colors=colors)
        scene.add_geometry(pct)
        pct.export(os.path.join(input_dir, "pointcloud.ply"))
        
    else:  
        X, C, X2, C2 = torch.load(path2, map_location=device)
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


    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

    # Add global frame
    global_frame = trimesh.creation.axis(origin_size=0.05, axis_length=0.5)
    scene.add_geometry(global_frame)

    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.show(line_settings={'point_size':1})
    #scene.export(file_obj=outfile)
    
    return outfile

def get_3D_model_from_scene(input_dir, silent, scene_state, camera_num=1, min_conf_thr=2, as_pointcloud=False, mask_sky=False, mask_floor=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    print("Outfile: ", outfile)

    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    masks = scene.masks
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    scale_factor = scene.get_scale_factor()
    quat_X = scene.get_quat_X()
    trans_X = scene.get_trans_X()

    ccam2pcam = scene.get_relative_poses()

    # Print estimated params
    if scale_factor is not None:
        print(f"Estimated scale factor: {scale_factor}")
        print(f"Estimated quaternion X: {quat_X}")
        print(f"Estimated scaled translation X: {scale_factor*trans_X}")

    relative_transformations = []
    output_camera_poses_dir = os.path.join(input_dir, os.path.join(f"camera{camera_num}", "estimated_camera_poses"))
    if os.path.exists(output_camera_poses_dir):
        shutil.rmtree(output_camera_poses_dir)

    os.makedirs(output_camera_poses_dir, exist_ok=True)
    
    for i in range(1, cams2world.shape[0]):
        T_prev_inv = torch.inverse(cams2world[i - 1])   # Inverse of previous transformation
        T_current = cams2world[i]                       # Current transformation
        T_rel = T_prev_inv @ T_current                  # Relative transformation
        relative_transformations.append(T_rel)

    # Save each relative transformation as a YAML file in OpenCV format
    for i, T_rel in enumerate(relative_transformations):
        # Convert PyTorch tensor to NumPy array
        T_rel_np = T_rel.numpy().astype(np.float64)  # OpenCV YAML format expects double precision (float64)

        # Define the output file name
        filename = os.path.join(output_camera_poses_dir, f"{i:04}.yaml")

        # Open a cv2.FileStorage for writing in YAML format
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("matrix", T_rel_np)
        fs.release()

        print(f"Saved relative transformation to {filename}")

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, pts3d_object, _, confs = to_numpy(tsdf.get_dense_pts3d(masks=masks, clean_depth=clean_depth))
    else:
        pts3d, pts3d_object, _, confs, confs_object = to_numpy(scene.get_dense_pts3d(masks=masks, clean_depth=clean_depth))

    msk = to_numpy([c > min_conf_thr for c in confs])
    msk_obj = to_numpy([c > min_conf_thr for c in confs_object])
    if scale_factor is not None:
        cams2world = ccam2pcam
    else:
        cams2world = cams2world
    print("Ready to convert scene to glb file...")
    return _convert_scene_output_to_glb(input_dir, camera_num, outfile, rgbimg, pts3d, pts3d_object, msk, msk_obj, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, silent=silent, mask_floor=mask_floor)


def get_reconstructed_scene(input_dir, outdir, config, scenegraph_type, model, device, filelist, mask_list, optim_level, intrinsic_params_vec, dist_coeffs, robot_poses, mask_floor, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    msks = None
    imgs, original_image_size = load_images_intr(filelist, config['image_size'], intrinsic_params_vec, verbose=not config['silent'])

    if len(mask_list)>0:    
        msks = load_masks(mask_list, filelist, intrinsic_params_vec, size=config['image_size'], verbose=not config['silent'])
    
    # Images include all the cameras 
    for cam, images in enumerate(imgs):
        print("Processing camera", cam+1)

        imgs = images
        msks_cam = msks[cam]
        intrinsic_params = intrinsic_params_vec[cam]
        dist_coeffs_cam = dist_coeffs[cam]
        print("Intrinsic parameters:", intrinsic_params)
        print("Distortion coefficients:", dist_coeffs_cam)
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
            filelist = [filelist[0], filelist[0] + '_2']

        scene_graph_params = [scenegraph_type]
        if scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(config['winsize']))
        elif scenegraph_type == "oneref":
            scene_graph_params.append(str(config['refid']))
        if scenegraph_type in ["swin", "logwin"] and not config['win_cyclic']:
            scene_graph_params.append('noncyclic')
        scene_graph = '-'.join(scene_graph_params)
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        if optim_level == 'coarse':
            config['niter2'] = 0
        
        # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
        if config['current_scene_state'] is not None and \
            not config['current_scene_state'].should_delete and \
                config['current_scene_state'].cache_dir is not None:
            cache_dir = config['current_scene_state'].cache_dir
        elif config['gradio_delete_cache']:
            cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
        else:
            cache_dir = os.path.join(outdir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        imagelist = filelist[cam]
        if robot_poses is not None:
            robot_poses_single = robot_poses[cam]
        else:
            robot_poses_single = None

        scene = sparse_global_alignment(imagelist, pairs, cache_dir,
                                        model, msks_cam, intrinsic_params=intrinsic_params, dist_coeffs_cam=dist_coeffs_cam, robot_poses=robot_poses_single, lr1=config['lr1'], niter1=config['niter1'], lr2=config['lr2'], niter2=config['niter2'], device=device,
                                        opt_depth='depth' in optim_level, shared_intrinsics=config['shared_intrinsics'],
                                        matching_conf_thr=config['matching_conf_thr'], **kw)
        
        if config['current_scene_state'] is not None and \
            not config['current_scene_state'].should_delete and \
                config['current_scene_state'].outfile_name is not None:
            outfile_name = config['current_scene_state'].outfile_name
        else:
            outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

        scene_state = SparseGAState(scene, config['gradio_delete_cache'], cache_dir, outfile_name)
        outfile = get_3D_model_from_scene(input_dir, config['silent'], scene_state, cam+1, config['min_conf_thr'], config['as_pointcloud'], config['mask_sky'], mask_floor,
                                        config['clean_depth'], config['transparent_cams'], config['cam_size'], config['TSDF_thresh'])

    return scene_state, outfile