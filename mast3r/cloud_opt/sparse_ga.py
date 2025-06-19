# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Sparse Global Alignement
# --------------------------------------------------------
from tqdm import tqdm
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import namedtuple
from functools import lru_cache
from scipy import sparse as sp
import copy
import time
from mast3r.utils.misc import mkdir_for, hash_md5
from mast3r.cloud_opt.utils.losses import gamma_loss
from mast3r.cloud_opt.utils.schedules import linear_schedule, cosine_schedule, cosine_schedule_with_restarts
from mast3r.fast_nn import fast_reciprocal_NNs, merge_corres

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import inv, geotrf  # noqa
from dust3r.utils.device import to_cpu, to_numpy, todevice  # noqa
from dust3r.post_process import estimate_focal_knowing_depth  # noqa
from dust3r.optim_factory import adjust_learning_rate_by_lr  # noqa
from dust3r.cloud_opt.base_opt import clean_pointcloud, clean_object_pointcloud
from dust3r.viz import SceneViz
from mast3r.utils.general_utils import reshape_list
from scipy.spatial.transform import Rotation
from utils.file_utils import *

PROCESS_ALL_IMAGES = "Multi-Camera"

class SparseGA():
    def __init__(self, img_paths, masks, corners_2d, pairs_in, res_fine, anchors, canonical_paths=None, scale_factor=None, trans_X=None, quat_X=None):
        def fetch_img(im):
            def torgb(x): return (x[0].permute(1, 2, 0).numpy() * .5 + .5).clip(min=0., max=1.)
            for im1, im2 in pairs_in:
                if im1['instance'] == im:
                    return torgb(im1['img'])
                if im2['instance'] == im:
                    return torgb(im2['img'])
        self.canonical_paths = canonical_paths
        self.img_paths = img_paths
        self.masks = masks
        self.corners_2d = corners_2d
        self.imgs = [fetch_img(img) for img in img_paths]
        self.intrinsics = res_fine['intrinsics']
        self.cam2w = res_fine['cam2w']
        self.depthmaps = res_fine['depthmaps']
        self.pts3d = res_fine['pts3d']
        self.pts3d_colors = []
        self.scale_factor = scale_factor
        self.quat_X = quat_X
        self.trans_X = trans_X
        self.working_device = self.cam2w.device
        for i in range(len(self.imgs)):
            im = self.imgs[i]
            x, y = anchors[i][0][..., :2].detach().cpu().numpy().T
            self.pts3d_colors.append(im[y, x])
            assert self.pts3d_colors[-1].shape == self.pts3d[i].shape
        self.n_imgs = len(self.imgs)

    def get_focals(self):
        return torch.tensor([ff[0, 0] for ff in self.intrinsics]).to(self.working_device)

    def get_principal_points(self):
        return torch.stack([ff[:2, -1] for ff in self.intrinsics]).to(self.working_device)

    def get_im_poses(self):
        return self.cam2w

    def get_relative_poses(self):

        # Inverse of the first camera pose


        cam2w_0_inv = torch.inverse(self.cam2w[0])

        # ee_to_rob = np.array([[9.99807842e-01, 9.69427142e-04, -1.95790426e-02, 5.55120952e-01],
        #                         [9.77803362e-04, -9.99999434e-01, 4.18246983e-04, 9.85213614e-05],
        #                         [-1.95786260e-02, -4.37311068e-04, -9.99808225e-01, 5.11640885e-01],
        #                         [0., 0., 0., 1.]])
        # cams2world_abs = [ee_to_rob @ h2e_list[0] @ cam_pose for cam_pose in cams2world]
        # cams2world = cams2world_abs
        # Compute relative poses
        ccam2pcam = [cam2w_0_inv @ cam2w_pose for cam2w_pose in self.cam2w]

        if self.scale_factor is not None:
            for i in range(len(ccam2pcam)):
                scale_idx = i // (len(ccam2pcam)//len(self.scale_factor))
                ccam2pcam[i][:3, 3] *= abs(self.scale_factor[scale_idx])
                # For robot arm scale cam2w 0 translation
                
                # ccam2pcam[i] = cam0@ccam2pcam[i]
        return ccam2pcam
    
    def get_scale_factor(self):
        return self.scale_factor

    def get_quat_X(self):
        return self.quat_X

    def get_trans_X(self):
        return self.trans_X

    def get_sparse_pts3d(self):
        return self.pts3d

    def get_dense_pts3d(self, masks=None, corners=None, pattern=None, clean_depth=True, subsample=8):
        assert self.canonical_paths, 'cache_path is required for dense 3d points'
        device = self.cam2w.device
        confs = []
        confs_object = []
        base_focals = []
        anchors = {}
        anchors_corners = {}
        all_masks_bool = []
        # H, W = masks[0].shape
        if isinstance(masks, list) and all(elem is None for elem in masks):
            masks = None

        all_conf_object = None
        if masks is not None:
            if masks[0] is not None:
                init_unique_masks = np.unique(masks[0])
            all_conf_object = [[None for _ in range(len(self.canonical_paths))] for _ in range(len(init_unique_masks))]
        
        for i, canon_path in enumerate(self.canonical_paths):
            (canon, canon2, conf), focal = torch.load(canon_path, map_location=device)
            confs.append(conf)
            H, W = conf.shape
            # Append object confidence (TODO: check if the masks is null)
            if masks is not None:
                if masks[i] is None:
                    masks[i] = np.zeros((H, W), dtype=int)
                
                unique_masks = np.unique(masks[i])
                for mask_value in unique_masks:
                    if mask_value == 0:
                        continue
                    mask_tensor = torch.from_numpy(masks[i]).to(conf.device)  # Convert mask to PyTorch tensor
                    mask_object = (mask_tensor == mask_value)
                    mask_bool = mask_object.bool()
                    conf_object = conf[mask_bool]
                    all_conf_object[mask_value-1][i] = conf_object
            base_focals.append(focal)

            pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
            idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
            anchors[i] = (pixels, idxs[i], offsets[i])  

            # For metric evaluation
            if corners is not None and corners[i] is not None:
                pixels_corners = torch.from_numpy(corners[i]).float().to(device)  # [N, 2] in formato (u, v)
                idxs_corners, offsets_corners = anchor_depth_offsets(canon2, {i: (pixels_corners, None)}, subsample=subsample)  # no subsampling per corner
                anchors_corners[i] = (pixels_corners, idxs_corners[i], offsets_corners[i])
                # print(f"Anchors corners {i}: {anchors_corners[i]}")

        # densify sparse depthmaps
        if masks is None:
            pts3d_object = None
            if self.scale_factor is None:
                pts3d, depthmaps = make_pts3d(anchors, self.intrinsics, self.cam2w, [
                                            d.ravel() for d in self.depthmaps], anchors_corners=anchors_corners, pattern=pattern, base_focals=base_focals, ret_depth=True)
                if clean_depth:
                    confs = clean_pointcloud(confs, self.intrinsics, inv(self.cam2w), depthmaps, pts3d)
            else:
                scale_factor_list = [self.scale_factor[i // int(len(base_focals)/len(self.scale_factor))] for i in range(len(base_focals))]
                pts3d, depthmaps = make_pts3d(anchors, self.intrinsics, self.get_relative_poses(), [
                                            d.ravel() for d in self.depthmaps], anchors_corners=anchors_corners, pattern=pattern, base_focals=base_focals, ret_depth=True, scale_factor=scale_factor_list)
                if clean_depth:
                    confs = clean_pointcloud(confs, self.intrinsics, inv(torch.stack(self.get_relative_poses(), dim=0)), depthmaps, pts3d)
        else:
            if self.scale_factor is None:
                pts3d, pts3d_object, depthmaps, depthmaps_obj = make_pts3d_mask(anchors, self.intrinsics, self.cam2w, [
                                            d.ravel() for d in self.depthmaps], masks=masks, anchors_corners=anchors_corners, pattern=pattern, base_focals=base_focals, ret_depth=True)
                clean_depth = False
                if clean_depth:
                    confs = clean_pointcloud(confs, self.intrinsics, inv(self.cam2w), depthmaps, pts3d)
            else:
                # Extract 3D points and 3D object points with respect to the first camera reference frame
                scale_factor_list = [self.scale_factor[i // int(len(base_focals)/len(self.scale_factor))] for i in range(len(base_focals))]
                pts3d, pts3d_object, depthmaps, depthmaps_obj  = make_pts3d_mask(anchors, self.intrinsics, self.get_relative_poses(), [
                                            d.ravel() for d in self.depthmaps], masks=masks, anchors_corners=anchors_corners, pattern=pattern, base_focals=base_focals, ret_depth=True, scale_factor=scale_factor_list)

                clean_depth = False
                if clean_depth:
                    confs = clean_pointcloud(confs, self.intrinsics, inv(torch.stack(self.get_relative_poses(), dim=0)), depthmaps, pts3d)

        return pts3d, pts3d_object, depthmaps, confs, all_conf_object

    def get_dense_original_pts3d(self, clean_depth=True, subsample=8):
        assert self.canonical_paths, 'cache_path is required for dense 3d points'
        device = self.cam2w.device
        confs = []
        base_focals = []
        anchors = {}
        for i, canon_path in enumerate(self.canonical_paths):
            (canon, canon2, conf), focal = torch.load(canon_path, map_location=device)
            confs.append(conf)
            base_focals.append(focal)

            H, W = conf.shape
            pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
            idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
            anchors[i] = (pixels, idxs[i], offsets[i])

        # densify sparse depthmaps
        pts3d, depthmaps = make_pts3d(anchors, self.intrinsics, self.cam2w, [
                                      d.ravel() for d in self.depthmaps], base_focals=base_focals, ret_depth=True)

        if clean_depth:
            confs = clean_pointcloud(confs, self.intrinsics, inv(self.cam2w), depthmaps, pts3d)

        return pts3d, depthmaps, confs

    def get_pts3d_colors(self):
        return self.pts3d_colors

    def get_depthmaps(self):
        return self.depthmaps

    def get_masks(self):
        return [slice(None, None) for _ in range(len(self.imgs))]

    def show(self, show_cams=True):
        pts3d, pts3d_object, _, confs = self.get_dense_pts3d()
        show_reconstruction(self.imgs, self.intrinsics if show_cams else None, self.cam2w,
                            [p.clip(min=-50, max=50) for p in pts3d],
                            masks=[c > 1 for c in confs])


def convert_dust3r_pairs_naming(imgs, pairs_in):
    for pair_id in range(len(pairs_in)):
        for i in range(2):
            pairs_in[pair_id][i]['instance'] = imgs[pairs_in[pair_id][i]['idx']]
    return pairs_in


def sparse_global_alignment(imgs, pairs_in, cache_path, model, opt_process=None, camera_num=None, masks = None, corners_2d=None, intrinsic_params=None, dist_coeffs_cam=None, robot_poses=None, multiple_camera_opt=None, subsample=8, desc_conf='desc_conf',
                            device='cuda', dtype=torch.float32, shared_intrinsics=False, **kw):
    """ Sparse alignment with MASt3R
        imgs: list of image paths
        cache_path: path where to dump temporary files (str)

        lr1, niter1: learning rate and #iterations for coarse global alignment (3D matching)
        lr2, niter2: learning rate and #iterations for refinement (2D reproj error)

        lora_depth: smart dimensionality reduction with depthmaps
    """
    
    # Convert pair naming convention from dust3r to mast3r
    pairs_in = convert_dust3r_pairs_naming(imgs, pairs_in)
    
    # forward pass
    pairs, cache_path = forward_mast3r(pairs_in, model,
                                       cache_path=cache_path, subsample=subsample,
                                       desc_conf=desc_conf, device=device)
    

    # extract canonical pointmaps    
    tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = \
            prepare_canonical_data(imgs, pairs, subsample, cache_path=cache_path, mode='avg-angle', device=device, camera_num=camera_num, intrinsic_params=intrinsic_params, opt_process=opt_process)

    # compute minimal spanning tree
    mst = compute_min_spanning_tree(pairwise_scores)

    # smartly combine all useful data
    imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21 = \
        condense_data(imgs, tmp_pairs, canonical_views, preds_21, dtype)
    # print("Base focals: ", base_focals)
    base_focals = base_focals.to(device)
    scale_factor, quat_X, trans_X = None, None, None
    # if intrinsic_params is None:
    print("CAMERA NUM: ", camera_num)
    imgs, res_fine, scale_factor, trans_X, quat_X = sparse_scene_optimizer(
        imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21, canonical_paths, mst,
        camera_num=camera_num, intrinsic_params=intrinsic_params, dist_coeffs_cam=dist_coeffs_cam, robot_poses=robot_poses, multiple_camera_opt=multiple_camera_opt, shared_intrinsics=shared_intrinsics, cache_path=cache_path, device=device, dtype=dtype, **kw)

    return SparseGA(imgs, masks, corners_2d, pairs_in, res_fine, anchors, canonical_paths, scale_factor, trans_X, quat_X)


def sparse_scene_optimizer(imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d,
                           preds_21, canonical_paths, mst, cache_path, camera_num=None, intrinsic_params=None, dist_coeffs_cam=None,
                           robot_poses=None, multiple_camera_opt=None,
                           lr1=0.2, niter1=500, loss1=gamma_loss(1.1),
                           lr2=0.02, niter2=500, loss2=gamma_loss(0.4),
                           lossd=gamma_loss(1.1),
                           opt_pp=True, opt_depth=True,
                           schedule=cosine_schedule, depth_mode='add', exp_depth=False,
                           lora_depth=False,  # dict(k=96, gamma=15, min_norm=.5),
                           shared_intrinsics=False,
                           init={}, device='cuda', dtype=torch.float32,
                           matching_conf_thr=5., loss_dust3r_w=0.01,
                           verbose=True, dbg=()):

    init = copy.deepcopy(init)
    # extrinsic parameters
    vec0001 = torch.tensor((0, 0, 0, 1), dtype=dtype, device=device)
    quats = [nn.Parameter(vec0001.clone()) for _ in range(len(imgs))]
    trans = [nn.Parameter(torch.zeros(3, device=device, dtype=dtype)) for _ in range(len(imgs))]

    # Initialization of hand-eye calibration parameters
    if robot_poses is not None:
        quat_X = [nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)) for _ in range(camera_num)]
        trans_X = [nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)) for _ in range(camera_num)]
        scale_factor = [nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype)) for _ in range(camera_num)]

    else:
        quat_X = None
        trans_X = None
        scale_factor = None

    # # If cameras are more than 1, we need to optimize their relative poses
    quat_X_rel = [nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)) for _ in range(camera_num) for _ in range(camera_num)]
    trans_X_rel = [nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)) for _ in range(camera_num) for _ in range(camera_num)]

    # initialize
    ones = torch.ones((len(imgs), 1), device=device, dtype=dtype)
    median_depths = torch.ones(len(imgs), device=device, dtype=dtype)
    for img in imgs:
        
        idx = imgs.index(img)
        init_values = init.setdefault(img, {})
        if verbose and init_values:
            print(f' >> initializing img=...{img[-25:]} [{idx}] for {set(init_values)}')

        K = init_values.get('intrinsics')
        if K is not None:
            K = K.detach()
            focal = K[:2, :2].diag().mean()
            pp = K[:2, 2]
            base_focals[idx] = focal
            pps[idx] = pp

        pps[idx] = pps[idx] / imsizes[idx]  # default principal_point would be (0.5, 0.5)

        depth = init_values.get('depthmap')
        if depth is not None:
            core_depth[idx] = depth.detach()

        median_depths[idx] = med_depth = core_depth[idx].median()
        core_depth[idx] /= med_depth

        cam2w = init_values.get('cam2w')
        if cam2w is not None:
            rot = cam2w[:3, :3].detach()
            cam_center = cam2w[:3, 3].detach()
            quats[idx].data[:] = roma.rotmat_to_unitquat(rot)
            trans_offset = med_depth * torch.cat((imsizes[idx] / base_focals[idx] * (0.5 - pps[idx]), ones[:1, 0]))
            trans[idx].data[:] = cam_center + rot @ trans_offset
            del rot
            assert False, 'inverse kinematic chain not yet implemented'
    
    canonical_paths_sep_cam = reshape_list(canonical_paths, camera_num)
    pps_sep_cam = reshape_list(pps, camera_num)
    imgs_sep_cam = reshape_list(imgs, camera_num)
    base_focals_sep_cam = reshape_list(base_focals, camera_num)
    core_depth_sep_cam = reshape_list(core_depth, camera_num)
    final_log_focals = []
    final_pps = []
    # intrinsics parameters
    if shared_intrinsics:
        # Optimize a single set of intrinsics for all cameras. Use averages as init.
        for cam_idx in range(camera_num): 
            confs_single_cam = torch.stack([torch.load(pth)[0][2].mean() for pth in canonical_paths_sep_cam[cam_idx]]).to(pps)
            weighting_single_cam = confs_single_cam / confs_single_cam.sum()
            pp_single_cam = nn.Parameter((weighting_single_cam @ pps_sep_cam[cam_idx]).to(dtype))
            if intrinsic_params is None:
                pps_sep_cam_sin = [pp_single_cam for _ in range(len(imgs_sep_cam[cam_idx]))]
                focal_m = weighting_single_cam @ base_focals_sep_cam[cam_idx]
                log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
                log_focals = [log_focal for _ in range(len(imgs_sep_cam[cam_idx]))]
                final_log_focals.append(log_focals)
                final_pps.append(pps_sep_cam_sin)
            else:
                pps_sep_cam_sin = [pp_single_cam.detach() for _ in range(len(imgs_sep_cam[cam_idx]))]
                focal_m = weighting_single_cam @ base_focals_sep_cam[cam_idx]
                log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
                log_focals = [log_focal.detach() for _ in range(len(imgs_sep_cam[cam_idx]))]
                final_log_focals.append(log_focals)
                final_pps.append(pps_sep_cam_sin)
            
    else:
        pps = [nn.Parameter(pp.to(dtype)) for pp in pps]
        log_focals = [nn.Parameter(f.view(1).log().to(dtype)) for f in base_focals]
        final_log_focals.append(log_focals)
        final_pps.append(pps)


    diags = imsizes.float().norm(dim=1)
    min_focals = 0.25 * diags  # diag = 1.2~1.4*max(W,H) => beta >= 1/(2*1.2*tan(fov/2)) ~= 0.26
    max_focals = 10 * diags


    imsizes_sep = reshape_list(imsizes, camera_num)
    min_focals_sep = reshape_list(min_focals, camera_num)
    max_focals_sep = reshape_list(max_focals, camera_num)

    assert len(mst[1]) == len(pps) - 1
    
    K_fixed_list = []
    focals_list = []
    distortion_list = []
    if intrinsic_params is not None:
        for cam_idx in range(camera_num):
        
            focals = torch.cat(final_log_focals[cam_idx]).exp().clip(
                min=min_focals_sep[cam_idx], max=max_focals_sep[cam_idx]
            )
            pps = torch.stack(final_pps[cam_idx])
            K_fixed = torch.eye(3, dtype=dtype, device=device)[None].expand(len(imgs_sep_cam[cam_idx]), 3, 3).clone()
            K_fixed[:, 0, 0] = K_fixed[:, 1, 1] = focals
            K_fixed[:, 0:2, 2] = pps * imsizes_sep[cam_idx]
            K_fixed = K_fixed.detach()
            K_fixed_list.append(K_fixed)
            focals_list.append(focals)
            repeated_distortion = torch.tensor(dist_coeffs_cam[cam_idx], dtype=dtype, device=device).repeat(len(imgs_sep_cam[cam_idx]), 1)
            distortion_list.append(repeated_distortion)
        focals_tensor = torch.cat(focals_list)
        K_fixed_tensor = torch.cat(K_fixed_list)
        distortion_tensor = torch.cat(distortion_list)


    flattened_log_focals = [tensor for inner_list in final_log_focals for tensor in inner_list]
    flattened_pps = [tensor for inner_list in final_pps for tensor in inner_list]

    def make_K_cam_depth_opt(log_focals, pps, trans, quats, log_sizes, core_depth):
        # make intrinsics
        focals = torch.cat(log_focals).exp().clip(min=min_focals, max=max_focals)
        pps = torch.stack(pps)
        K = torch.eye(3, dtype=dtype, device=device)[None].expand(len(imgs), 3, 3).clone()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, 0:2, 2] = pps * imsizes
        if trans is None:
            return K

        # security! optimization is always trying to crush the scale down
        sizes = torch.cat(log_sizes).exp()
        global_scaling = 1 / sizes.min()

        # compute distance of camera to focal plane
        # tan(fov) = W/2 / focal

        z_cameras = sizes * median_depths * focals / base_focals

        # make extrinsic
        rel_cam2cam = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4).clone()
        rel_cam2cam[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats), dim=1))
        rel_cam2cam[:, :3, 3] = torch.stack(trans)

        # camera are defined as a kinematic chain
        tmp_cam2w = [None] * len(K)
        tmp_cam2w[mst[0]] = rel_cam2cam[mst[0]]
        for i, j in mst[1]:
            # i is the cam_i_to_world reference, j is the relative pose = cam_j_to_cam_i
            tmp_cam2w[j] = tmp_cam2w[i] @ rel_cam2cam[j]
        tmp_cam2w = torch.stack(tmp_cam2w)

        # smart reparameterizaton of cameras
        trans_offset = z_cameras.unsqueeze(1) * torch.cat((imsizes / focals.unsqueeze(1) * (0.5 - pps), ones), dim=-1)
        new_trans = global_scaling * (tmp_cam2w[:, :3, 3:4] - tmp_cam2w[:, :3, :3] @ trans_offset.unsqueeze(-1))
        cam2w = torch.cat((torch.cat((tmp_cam2w[:, :3, :3], new_trans), dim=2),
                          vec0001.view(1, 1, 4).expand(len(K), 1, 4)), dim=1)

        depthmaps = []
        for i in range(len(imgs)):
            core_depth_img = core_depth[i]
            if exp_depth:
                core_depth_img = core_depth_img.exp()
            if lora_depth:  # compute core_depth as a low-rank decomposition of 3d points
                core_depth_img = lora_depth_proj[i] @ core_depth_img
            if depth_mode == 'add':
                core_depth_img = z_cameras[i] + (core_depth_img - 1) * (median_depths[i] * sizes[i])
            elif depth_mode == 'mul':
                core_depth_img = z_cameras[i] * core_depth_img
            else:
                raise ValueError(f'Bad {depth_mode=}')
            depthmaps.append(global_scaling * core_depth_img)

        return K, (inv(cam2w), cam2w), depthmaps
    
    def make_K_cam_depth(K_fixed_tensor, pps, trans, quats, log_sizes, core_depth):
        if trans is None:
            return K_fixed_tensor

        # security! optimization is always trying to crush the scale down
        sizes = torch.cat(log_sizes).exp()
        global_scaling = 1 / sizes.min()

        # compute distance of camera to focal plane
        # tan(fov) = W/2 / focal

        z_cameras = sizes * median_depths * focals_tensor / base_focals

        # make extrinsic
        rel_cam2cam = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4).clone()
        rel_cam2cam[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats), dim=1))
        rel_cam2cam[:, :3, 3] = torch.stack(trans)

        # camera are defined as a kinematic chain
        tmp_cam2w = [None] * len(K_fixed_tensor)
        tmp_cam2w[mst[0]] = rel_cam2cam[mst[0]]
        for i, j in mst[1]:
            # i is the cam_i_to_world reference, j is the relative pose = cam_j_to_cam_i
            tmp_cam2w[j] = tmp_cam2w[i] @ rel_cam2cam[j]
        tmp_cam2w = torch.stack(tmp_cam2w)

        # smart reparameterizaton of cameras
        pps_tensor = torch.stack(pps) 
        trans_offset = z_cameras.unsqueeze(1) * torch.cat((imsizes / focals_tensor.unsqueeze(1) * (0.5 - pps_tensor), ones), dim=-1)
        new_trans = global_scaling * (tmp_cam2w[:, :3, 3:4] - tmp_cam2w[:, :3, :3] @ trans_offset.unsqueeze(-1))
        cam2w = torch.cat((torch.cat((tmp_cam2w[:, :3, :3], new_trans), dim=2),
                          vec0001.view(1, 1, 4).expand(len(K_fixed_tensor), 1, 4)), dim=1)

        depthmaps = []
        for i in range(len(imgs)):
            core_depth_img = core_depth[i]
            if exp_depth:
                core_depth_img = core_depth_img.exp()
            if lora_depth:  # compute core_depth as a low-rank decomposition of 3d points
                core_depth_img = lora_depth_proj[i] @ core_depth_img
            if depth_mode == 'add':
                core_depth_img = z_cameras[i] + (core_depth_img - 1) * (median_depths[i] * sizes[i])
            elif depth_mode == 'mul':
                core_depth_img = z_cameras[i] * core_depth_img
            else:
                raise ValueError(f'Bad {depth_mode=}')
            depthmaps.append(global_scaling * core_depth_img)

        return K_fixed_tensor, (inv(cam2w), cam2w), depthmaps
    
    

    core_depth_list = []
    log_sizes_list = []
    
    K = make_K_cam_depth_opt(flattened_log_focals, flattened_pps, None, None, None, None)
    K_sep_cam = reshape_list(K, camera_num)
    for cam_idx in range(camera_num):
        if intrinsic_params is None:
            if shared_intrinsics:
                print(f'init focal camera {cam_idx + 1} (shared) = {to_numpy(K_sep_cam[cam_idx][0, 0, 0]).round(2)}')
            else:
                print('init focals =', to_numpy(K_sep_cam[cam_idx][:, 0, 0]))
        else:
            if shared_intrinsics:
                print(f'init focal camera {cam_idx + 1} (shared) = {to_numpy(K_fixed_list[cam_idx][0, 0, 0]).round(2)}')
            else:
                print('init focals =', to_numpy(K_fixed_list[cam_idx][:, 0, 0]))

        # spectral low-rank projection of depthmaps
        if lora_depth:
            if intrinsic_params is None:
                core_depth_sep_cam[cam_idx], lora_depth_proj = spectral_projection_of_depthmaps(
                    imgs_sep_cam[cam_idx], K_sep_cam[cam_idx], core_depth_sep_cam[cam_idx], subsample, cache_path=cache_path, **lora_depth)
            else:
                core_depth_sep_cam[cam_idx], lora_depth_proj = spectral_projection_of_depthmaps(
                    imgs_sep_cam[cam_idx], K_fixed_list[cam_idx], core_depth_sep_cam[cam_idx], subsample, cache_path=cache_path, **lora_depth)
        if exp_depth:
            core_depth_sep_cam[cam_idx] = [d.clip(min=1e-4).log() for d in core_depth_sep_cam[cam_idx]]
        core_depth_sep_cam[cam_idx] = [nn.Parameter(d.ravel().to(dtype)) for d in core_depth_sep_cam[cam_idx]]
        log_sizes = [nn.Parameter(torch.zeros(1, dtype=dtype, device=device)) for _ in range(len(imgs_sep_cam[cam_idx]))]
        core_depth_list.append(core_depth_sep_cam[cam_idx])
        log_sizes_list.append(log_sizes)

    flattened_core_depth = [tensor for inner_list in core_depth_list for tensor in inner_list]
    flattened_log_sizes = [tensor for inner_list in log_sizes_list for tensor in inner_list]

    # Fetch img slices
    _, confs_sum, imgs_slices = corres
    # Define which pairs are fine to use with matching
    def matching_check(x): return x.max() > matching_conf_thr
    is_matching_ok = {}
    for s in imgs_slices:
        is_matching_ok[s.img1, s.img2] = matching_check(s.confs)

    # Prepare slices and corres for losses
    dust3r_slices = [s for s in imgs_slices if not is_matching_ok[s.img1, s.img2]]
    loss3d_slices = [s for s in imgs_slices if is_matching_ok[s.img1, s.img2]]
    cleaned_corres2d = []
    for cci, (img1, pix1, confs, confsum, imgs_slices) in enumerate(corres2d):
        cf_sum = 0
        pix1_filtered = []
        confs_filtered = []
        curstep = 0
        cleaned_slices = []
        for img2, slice2 in imgs_slices:
            if is_matching_ok[img1, img2]:
                tslice = slice(curstep, curstep + slice2.stop - slice2.start, slice2.step)
                pix1_filtered.append(pix1[tslice])
                confs_filtered.append(confs[tslice])
                cleaned_slices.append((img2, slice2))
            curstep += slice2.stop - slice2.start
        if pix1_filtered != []:
            pix1_filtered = torch.cat(pix1_filtered)
            confs_filtered = torch.cat(confs_filtered)
            cf_sum = confs_filtered.sum()
        cleaned_corres2d.append((img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices))

    def loss_dust3r(cam2w, pts3d, pix_loss):
        # In the case no correspondence could be established, fallback to DUSt3R GA regression loss formulation (sparsified)
        loss = 0.
        cf_sum = 0.
        for s in dust3r_slices:
            if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                continue
            # fallback to dust3r regression
            tgt_pts, tgt_confs = preds_21[imgs[s.img2]][imgs[s.img1]]
            tgt_pts = geotrf(cam2w[s.img2], tgt_pts)
            cf_sum += tgt_confs.sum()
            loss += tgt_confs @ pix_loss(pts3d[s.img1], tgt_pts)
        return loss / cf_sum if cf_sum != 0. else 0.

    def loss_3d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        if any(v.get('freeze') for v in init.values()):
            pts3d_1 = []
            pts3d_2 = []
            confs = []
            for s in loss3d_slices:
                if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                    continue
                pts3d_1.append(pts3d[s.img1][s.slice1])
                pts3d_2.append(pts3d[s.img2][s.slice2])
                confs.append(s.confs)
        else:
            pts3d_1 = [pts3d[s.img1][s.slice1] for s in loss3d_slices]
            pts3d_2 = [pts3d[s.img2][s.slice2] for s in loss3d_slices]
            confs = [s.confs for s in loss3d_slices]

        if pts3d_1 != []:
            confs = torch.cat(confs)
            pts3d_1 = torch.cat(pts3d_1)
            pts3d_2 = torch.cat(pts3d_2)
            loss = confs @ pix_loss(pts3d_1, pts3d_2)
            cf_sum = confs.sum()
        else:
            loss = 0.
            cf_sum = 1.
        return loss / cf_sum

    def loss_2d_K_opt(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        proj_matrix = K @ w2cam[:, :3]
        loss = npix = 0
        for img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices in cleaned_corres2d:
            if init[imgs[img1]].get('freeze', 0) >= 1:
                continue  # no need
            pts3d_in_img1 = [pts3d[img2][slice2] for img2, slice2 in cleaned_slices]
            if pts3d_in_img1 != []:
                pts3d_in_img1 = torch.cat(pts3d_in_img1)
                loss += confs_filtered @ pix_loss(pix1_filtered, reproj2d(proj_matrix[img1], pts3d_in_img1))
                npix += confs_filtered.sum()

        return loss / npix if npix != 0 else 0.
    
    def loss_2d(K_fixed, dist_coeffs_cam, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        #proj_matrix = K_fixed @ w2cam[:, :3]

        proj_matrix_w_dist = w2cam[:, :3]
        loss = npix = 0
        for img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices in cleaned_corres2d:
            if init[imgs[img1]].get('freeze', 0) >= 1:
                continue  # no need
            pts3d_in_img1 = [pts3d[img2][slice2] for img2, slice2 in cleaned_slices]
            if pts3d_in_img1 != []:
                pts3d_in_img1 = torch.cat(pts3d_in_img1)
                #proj_points = reproj2d(proj_matrix[img1], pts3d_in_img1)
                proj_points_w_dist = reproj2d_with_dist(proj_matrix_w_dist[img1], pts3d_in_img1, K_fixed[img1], dist_coeffs_cam[img1])
                
                valid_mask = torch.all(torch.abs(proj_points_w_dist) < 1e6, dim=1)  # Threshold: 1e6
                if not valid_mask.any():
                    print(f"Skipping image {img1} due to all invalid projections.")
                    continue
                pix1_filtered = pix1_filtered[valid_mask]
                proj_points_w_dist = proj_points_w_dist[valid_mask]
                confs_filtered = confs_filtered[valid_mask]
                pixel_loss_component = pix_loss(pix1_filtered, proj_points_w_dist)

                loss_component = confs_filtered @ pixel_loss_component
                if torch.isinf(loss_component):
                    print(f"Image {img1} caused inf in loss_component.")
                    continue  # Skip this image
                loss += loss_component
                npix += confs_filtered.sum()

        return loss / npix if npix != 0 else 0.
    
    def calibration_loss(w2cam, robot_poses, scale_factor, quat_X, trans_X, quat_X_rel, trans_X_rel):
        """
        Loss function exploiting robot kinematics with quaternion rotation representation.
        """
        loss = 0.0
        X_list = [None] * len(scale_factor)
        # Ensure tensors are on the correct device and dtype
        device = w2cam[0].device
        dtype = w2cam[0].dtype  

        # Compute camera extrinsics    
        for i in range(len(scale_factor)):
            X_rot = quaternion_to_matrix(quat_X[i])
            X = torch.cat([torch.cat([X_rot, trans_X[i].view(3, 1)], dim=1), 
                        torch.tensor([[0, 0, 0, 1]], device=device, dtype=dtype)], dim=0)
            X_list[i] = X


        # Compute the rotation magnitude
        rotation_magnitude_list = []
        wcam_reshaped = reshape_list(w2cam, camera_num)
        A_List, B_List = [], []
    
        for cam_idx in range(camera_num):
            A_cam, B_cam = [], []
            for i in range(1, len(wcam_reshaped[cam_idx])):
                A = robot_poses[i - 1]
                B = wcam_reshaped[cam_idx][i - 1] @ torch.linalg.inv(wcam_reshaped[cam_idx][i]) 
                A_cam.append(A)
                B_cam.append(B)
                angle_axis = matrix_to_axis_angle(A[:3, :3])
                rotation_magnitude_list.append(torch.norm(angle_axis, dim=0))
            A_List.append(A_cam)
            B_List.append(B_cam)

        max_val = max(rotation_magnitude_list)
        min_val = min(rotation_magnitude_list)
        rotation_magnitude_list = [(val - min_val) / (max_val - min_val) for val in rotation_magnitude_list]
        rotation_magnitude_list = [val**2 for val in rotation_magnitude_list]
        
        for cam_idx in range(camera_num):
            for i in range(1, len(wcam_reshaped[cam_idx])):
                A = A_List[cam_idx][i - 1]
                B = B_List[cam_idx][i - 1]
                
                A = A.to(device).to(dtype)
                B = B.to(device).to(dtype)
                B_rotated = B.clone()
                chain1 = A
                chain2 = X_list[cam_idx] @ B_rotated @ torch.linalg.inv(X_list[cam_idx])
            
                # Scale the translation part of chain2
                # chain2 = chain2.clone()
                chain2[:3, 3] *= torch.abs(scale_factor[cam_idx])

                # Compute rotation loss
                chain1_quat = matrix_to_quaternion(chain1[:3, :3])
                chain2_quat = matrix_to_quaternion(chain2[:3, :3])
                rotation_magnitude = rotation_magnitude_list[i - 1]
                rotation_loss = rotation_magnitude * torch.nn.functional.mse_loss(chain1_quat, chain2_quat)

                # Compute translation loss
                translation_loss = rotation_magnitude * torch.nn.functional.mse_loss(chain1[:3, 3], chain2[:3, 3])

                # Combine losses
                loss += rotation_loss + translation_loss


        if multiple_camera_opt:
            # Enforce Relative Pose Consistency Across Time
            for t in range(1, len(wcam_reshaped[0])):  # Iterate over time steps

                for cam_i in range(camera_num):
                    for cam_j in range(cam_i + 1, camera_num):
                        # Compute relative transformation T_{i,j}^{(t)}
                        T_i_world = wcam_reshaped[cam_i][t]
                        T_j_world = wcam_reshaped[cam_j][t]
                        # X_ij_t = torch.linalg.inv(T_i_world) @ T_j_world  # Relative pose at time t
                        X_rot_rel = quaternion_to_matrix(quat_X_rel[cam_i * camera_num + cam_j])
                        X_rel = torch.cat([torch.cat([X_rot_rel, trans_X_rel[cam_i * camera_num + cam_j].view(3, 1)], dim=1), 
                                    torch.tensor([[0, 0, 0, 1]], device=device, dtype=dtype)], dim=0)
                        
                        # Compute relative transformation T_{i,j}^{(t-1)}
                        T_i_world_prev = wcam_reshaped[cam_i][t - 1]
                        T_j_world_prev = wcam_reshaped[cam_j][t - 1]
                        # X_ij_prev = torch.linalg.inv(T_i_world_prev) @ T_j_world_prev  # Relative pose at t-1
                        B_i = torch.linalg.inv(T_i_world_prev) @ T_i_world
                        B_j = torch.linalg.inv(T_j_world_prev) @ T_j_world
                        
                        chain1 = B_i @ X_rel
                        chain2 = X_rel @ B_j
                        # Compute difference in relative transformations
                        rel_rot_t = chain1[:3, :3]
                        rel_rot_prev = chain2[:3, :3]
                        rel_trans_t = chain1[:3, 3]
                        rel_trans_prev = chain2[:3, 3]

                        # Convert rotation matrices to quaternions
                        rel_quat_t = matrix_to_quaternion(rel_rot_t)
                        rel_quat_prev = matrix_to_quaternion(rel_rot_prev)

                        # Compute consistency loss
                        rel_rotation_loss = torch.nn.functional.mse_loss(rel_quat_t, rel_quat_prev)
                        rel_translation_loss = torch.nn.functional.mse_loss(rel_trans_t, rel_trans_prev)

                        # Add to total loss
                        loss += rel_rotation_loss + rel_translation_loss
                    
        return loss
    
    def optimize_loop_with_calibration_and_2d(
        loss_3d_func,
        loss_2d_func,
        lr_base,
        niter,
        pix_loss,
        calibration_loss_func,
        lr_end=0,
        dynamic_weights=True
    ):
        """
        Unified optimization loop integrating 3D loss, 2D reprojection loss, and calibration loss.
        """
        # Create separate optimizers for different parameter sets
        camera_params = quats + trans + flattened_log_sizes
        calibration_params = scale_factor + quat_X + trans_X + quat_X_rel + trans_X_rel

        optimizer_camera = torch.optim.Adam(camera_params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        optimizer_calibration = torch.optim.Adam(calibration_params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        
        ploss = pix_loss if 'meta' in repr(pix_loss) else (lambda a: pix_loss)

        with tqdm(total=niter) as bar:
            for iter in range(niter or 1):
                # Compute camera poses and points 
                _, (w2cam, cam2w), depthmaps = make_K_cam_depth(K_fixed_tensor, flattened_pps, trans, quats, flattened_log_sizes, flattened_core_depth)
                pts3d = make_pts3d(anchors, K_fixed_tensor, cam2w, depthmaps, base_focals=base_focals)
                if niter == 0:
                    break

                # Adjust learning rate
                alpha = (iter / niter) # It increases over time

                if dynamic_weights:
                    weight_2d = 1.0 * (1 - alpha)  # Decrease over time
                    weight_calib = 1.0 + alpha  # Increase over time
                    weight_3d = 1.0  # Keep constant
                else:
                    weight_2d = 1.0
                    weight_calib = 1.0
                    weight_3d = 1.0

                lr = schedule(alpha, lr_base, lr_end)
                adjust_learning_rate_by_lr(optimizer_camera, lr)
                adjust_learning_rate_by_lr(optimizer_calibration, lr)  # Lower learning rate for calibration optimizer

                # pix_loss = ploss(1 - alpha)

                optimizer_camera.zero_grad()
                optimizer_calibration.zero_grad()

                # Compute individual losses
                reprojection_loss = loss_2d_func(K_fixed_tensor, distortion_tensor, w2cam, pts3d, pix_loss) + loss_dust3r_w * loss_dust3r(cam2w, pts3d, lossd)
                calib_loss = calibration_loss_func(w2cam, robot_poses, scale_factor, quat_X, trans_X, quat_X_rel, trans_X_rel)
                loss_3d = loss_3d_func(K_fixed_tensor, w2cam, pts3d, pix_loss)

                # Weighted total loss
                total_loss = (weight_2d * reprojection_loss +
                            weight_calib * calib_loss + weight_3d * loss_3d)

                # Backpropagation and optimization
                total_loss.backward()
                optimizer_camera.step()
                optimizer_calibration.step()

                # Normalize quaternions
                for i in range(len(imgs)):
                    quats[i].data[:] /= quats[i].data.norm()
                for i in range(len(scale_factor)):
                    quat_X[i].data = quat_X[i].data / quat_X[i].data.norm()

                # Check for NaN or other optimization issues
                loss = float(total_loss)
                if loss != loss:  # NaN loss
                    break

                # Progress bar update
                bar.set_postfix_str(f'{lr=:.4f}, {loss=:.3f}')
                bar.update(1)

        if niter:
            print(f'>> final loss = {loss}')
        return dict(
            intrinsics=K_fixed_tensor,
            cam2w=cam2w.detach(),
            depthmaps=[d.detach() for d in depthmaps],
            pts3d=[p.detach() for p in pts3d]
        )

    def optimize_loop_K_opt(loss_func, lr_base, niter, pix_loss, lr_end=0):
        # create optimizer
        params = flattened_pps + flattened_log_focals + quats + trans + flattened_log_sizes + flattened_core_depth
        optimizer = torch.optim.Adam(params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        ploss = pix_loss if 'meta' in repr(pix_loss) else (lambda a: pix_loss)

        with tqdm(total=niter) as bar:
            for iter in range(niter or 1):
                K, (w2cam, cam2w), depthmaps = make_K_cam_depth_opt(flattened_log_focals, flattened_pps, trans, quats, flattened_log_sizes, flattened_core_depth)
                pts3d = make_pts3d(anchors, K, cam2w, depthmaps, base_focals=base_focals)
                if niter == 0:
                    break

                alpha = (iter / niter)
                lr = schedule(alpha, lr_base, lr_end)
                adjust_learning_rate_by_lr(optimizer, lr)
                pix_loss = ploss(1 - alpha)
                optimizer.zero_grad()
                loss = loss_func(K, w2cam, pts3d, pix_loss) + loss_dust3r_w * loss_dust3r(cam2w, pts3d, lossd)
                loss.backward()
                optimizer.step()

                # make sure the pose remains well optimizable
                for i in range(len(imgs)):
                    quats[i].data[:] /= quats[i].data.norm()

                loss = float(loss)
                if loss != loss:
                    break  # NaN loss
                bar.set_postfix_str(f'{lr=:.4f}, {loss=:.3f}')
                bar.update(1)

        if niter:
            print(f'>> final loss = {loss}')
        return dict(intrinsics=K.detach(), cam2w=cam2w.detach(),
                    depthmaps=[d.detach() for d in depthmaps], pts3d=[p.detach() for p in pts3d])
    
    def optimize_loop(loss_func, lr_base, niter, pix_loss, lr_end=0):
        # create optimizer
        params = quats + trans + flattened_log_sizes + flattened_core_depth
        optimizer = torch.optim.Adam(params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        ploss = pix_loss if 'meta' in repr(pix_loss) else (lambda a: pix_loss)

        with tqdm(total=niter) as bar:
            for iter in range(niter or 1):
                _, (w2cam, cam2w), depthmaps = make_K_cam_depth(K_fixed_tensor, flattened_pps, trans, quats, flattened_log_sizes, flattened_core_depth)
                pts3d = make_pts3d(anchors, K_fixed_tensor, cam2w, depthmaps, base_focals=base_focals)
                if niter == 0:
                    break

                alpha = (iter / niter)
                lr = schedule(alpha, lr_base, lr_end)
                adjust_learning_rate_by_lr(optimizer, lr)
                pix_loss = ploss(1 - alpha)
                optimizer.zero_grad()
                loss = loss_func(K_fixed_tensor, w2cam, pts3d, pix_loss) + loss_dust3r_w * loss_dust3r(cam2w, pts3d, lossd)
                loss.backward()
                optimizer.step()

                # make sure the pose remains well optimizable
                for i in range(len(imgs)):
                    quats[i].data[:] /= quats[i].data.norm()

                loss = float(loss)
                if loss != loss:
                    break  # NaN loss
                bar.set_postfix_str(f'{lr=:.4f}, {loss=:.3f}')
                bar.update(1)

        if niter:
            print(f'>> final loss = {loss}')
        return dict(intrinsics=K_fixed_tensor, cam2w=cam2w.detach(),
                    depthmaps=[d.detach() for d in depthmaps], pts3d=[p.detach() for p in pts3d])


    # If robot poses are not provided, optimize only 3D points without calibration
    if robot_poses is None:
        # at start, don't optimize 3d points
        for i, img in enumerate(imgs):
            trainable = not (init[img].get('freeze'))
            if intrinsic_params is None:
                flattened_pps[i].requires_grad_(False)
                flattened_log_focals[i].requires_grad_(False)
            else: 
                flattened_pps[i] = flattened_pps[i].detach()
                flattened_log_focals[i] = flattened_log_focals[i].detach()

            quats[i].requires_grad_(trainable)
            trans[i].requires_grad_(trainable)
            flattened_log_sizes[i].requires_grad_(trainable)
            flattened_core_depth[i].requires_grad_(False)

        # log_focals = torch.tensor(log_focals, dtype=pps.dtype, device=pps.device)

        if intrinsic_params is not None:
            res_coarse = optimize_loop(
                lambda _, w2cam, pts3d, pix_loss: loss_3d(K_fixed_tensor, w2cam, pts3d, pix_loss),
                lr_base=lr1,
                niter=niter1,
                pix_loss=loss1,
            )
        else:
            res_coarse = optimize_loop_K_opt(loss_3d, lr_base=lr1, niter=niter1, pix_loss=loss1)

        res_fine = None
        if niter2:
            # now we can optimize 3d points
            for i, img in enumerate(imgs):
                if init[img].get('freeze', 0) >= 1:
                    continue
                if intrinsic_params is None:
                    flattened_pps[i].requires_grad_(bool(opt_pp))
                    flattened_log_focals[i].requires_grad_(True)
                    flattened_core_depth[i].requires_grad_(True)

            # refinement with 2d reproj
            if intrinsic_params is not None:
                # res_fine = optimize_loop(loss_2d, lr_base=lr2, niter=niter2, pix_loss=loss2)
                res_fine = optimize_loop(
                    lambda _, w2cam, pts3d, pix_loss: loss_2d(K_fixed_tensor, distortion_tensor, w2cam, pts3d, pix_loss),
                    lr_base=lr2,
                    niter=niter2,
                    pix_loss=loss2
                )
            else:
                res_fine = optimize_loop_K_opt(loss_2d_K_opt, lr_base=lr2, niter=niter2, pix_loss=loss2)
        
        if intrinsic_params is None:
            K = make_K_cam_depth_opt(flattened_log_focals, flattened_pps, None, None, None, None)
            K_cam_sep = reshape_list(K, camera_num)
            for cam_idx in range(camera_num):
                if shared_intrinsics:
                    print(f'Final focal camera {cam_idx + 1} (shared) = {to_numpy(K_cam_sep[cam_idx][0, 0, 0]).round(2)}')
                else:
                    print(f'Final focals camera {cam_idx + 1} = {to_numpy(K_cam_sep[cam_idx][:, 0, 0])}')
        else:
            K_cam_fixed_sep = reshape_list(K_fixed_tensor, camera_num)
            for cam_idx in range(camera_num):
                if shared_intrinsics:
                    print('Final focal (shared) = ', to_numpy(K_cam_fixed_sep[cam_idx][0, 0, 0]).round(2))
                else:
                    print('Final focals =', to_numpy(K_cam_fixed_sep[cam_idx][:, 0, 0]))

        return imgs, res_fine, scale_factor, trans_X, quat_X

    else:
        if intrinsic_params is not None:
            # at start, don't optimize 3d points
            for i, img in enumerate(imgs):
                trainable = not (init[img].get('freeze'))
                flattened_pps[i] = flattened_pps[i].detach()
                flattened_log_focals[i] = flattened_log_focals[i].detach()

                quats[i].requires_grad_(trainable)
                trans[i].requires_grad_(trainable)
                flattened_log_sizes[i].requires_grad_(trainable)
                
            for i in range(camera_num):
                quat_X[i].requires_grad_(True)
                trans_X[i].requires_grad_(True)
                scale_factor[i].requires_grad_(True)
                for j in range(camera_num):
                    quat_X_rel[i][j].requires_grad_(True)
                    trans_X_rel[i][j].requires_grad_(True)

            res_fine = None
            res_fine = optimize_loop_with_calibration_and_2d(
                lambda _, w2cam, pts3d, pix_loss: loss_3d(K_fixed_tensor, w2cam, pts3d, pix_loss),
                lambda K_fixed_tensor, distortion_tensor, w2cam, pts3d, pix_loss: loss_2d(K_fixed_tensor, distortion_tensor, w2cam, pts3d, pix_loss),
                lr_base=lr1,
                niter=niter1,
                pix_loss=loss2,
                calibration_loss_func=calibration_loss,
                dynamic_weights=True
            )

            if shared_intrinsics:
                print('Final focal (shared) = ', to_numpy(K_fixed[0, 0, 0]).round(2))
            else:
                print('Final focals =', to_numpy(K_fixed[:, 0, 0]))

        else:
            print("ATTENTION: Intrinsic params must be provided.")
        return imgs, res_fine, scale_factor, trans_X, quat_X



@lru_cache
def mask110(device, dtype):
    return torch.tensor((1, 1, 0), device=device, dtype=dtype)


def proj3d(inv_K, pixels, z):
    if pixels.shape[-1] == 2:
        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    return z.unsqueeze(-1) * (pixels * inv_K.diag() + inv_K[:, 2] * mask110(z.device, z.dtype))

def checkerboard_square_stats(all_pts3d_corners, pattern_size):
    all_dists = []

    num_cols, num_rows = pattern_size  # Es. (9, 6)

    for corners3d in all_pts3d_corners:
        if corners3d.shape[0] != num_cols * num_rows:
            print("Warning: unexpected number of corners.")
            continue

        corners3d = corners3d.detach().cpu().numpy() if hasattr(corners3d, 'cpu') else corners3d

        # Reshape i corner in (rows, cols, 3)
        grid = corners3d.reshape((num_rows, num_cols, 3))

        # Distanze tra colonne (orizzontale)
        dx = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2)  # shape: (num_rows, num_cols-1)
        # Distanze tra righe (verticale)
        dy = np.linalg.norm(grid[1:, :] - grid[:-1, :], axis=2)  # shape: (num_rows-1, num_cols)

        all_dists.extend(dx.flatten())
        all_dists.extend(dy.flatten())

    all_dists = np.array(all_dists)
    stats = {
        'mean': np.mean(all_dists),
        'std': np.std(all_dists),
        'min': np.min(all_dists),
        'max': np.max(all_dists),
        'num_samples': len(all_dists)
    }

    return stats

def make_pts3d(anchors, K, cam2w, depthmaps, anchors_corners=None, pattern=None, base_focals=None, ret_depth=False, scale_factor=None):
    focals = K[:, 0, 0]
    invK = inv(K)
    all_pts3d = []
    depth_out = []
    all_pts3d_corners = []
    depth_out_corners = []
    # print("ANCHORS CORNERS: ", anchors_corners)
    for img, (pixels, idxs, offsets) in anchors.items():
        
        # from depthmaps to 3d points
        if base_focals is None:
            pass
        else:
            # compensate for focal
            # depth + depth * (offset - 1) * base_focal / focal
            # = depth * (1 + (offset - 1) * (base_focal / focal))
            if scale_factor is None:
                offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])
            else:
                offsets = (1 + (offsets - 1) * (base_focals[img] / focals[img]))*scale_factor[img]
                
        pts3d = proj3d(invK[img], pixels, depthmaps[img][idxs] * offsets)
        if ret_depth:
            depth_out.append(pts3d[..., 2])  # before camera rotation

        # rotate to world coordinate
        pts3d = geotrf(cam2w[img], pts3d)
        all_pts3d.append(pts3d)

    if anchors_corners is not None:
        for img, (pixels, idxs, offsets) in anchors_corners.items():
            
            # from depthmaps to 3d points
            if base_focals is None:
                pass
            else:
                # compensate for focal
                # depth + depth * (offset - 1) * base_focal / focal
                # = depth * (1 + (offset - 1) * (base_focal / focal))
                if scale_factor is None:
                    offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])
                else:
                    offsets = (1 + (offsets - 1) * (base_focals[img] / focals[img]))*scale_factor[img]
                    
            pts3d = proj3d(invK[img], pixels, depthmaps[img][idxs] * offsets)
            if ret_depth:
                depth_out_corners.append(pts3d[..., 2])  # before camera rotation

            # rotate to world coordinate
            pts3d = geotrf(cam2w[img], pts3d)
            all_pts3d_corners.append(pts3d)
        
        if len (all_pts3d_corners) > 0:
            stats = checkerboard_square_stats(all_pts3d_corners, pattern)

            print("Checkerboard square statistics:")
            for k, v in stats.items():
                print(f"{k}: {v:.4f}")


    if ret_depth:
        return all_pts3d, depth_out
    return all_pts3d


def make_pts3d_mask(anchors, K, cam2w, depthmaps, masks=None, anchors_corners=None, pattern=None, base_focals=None, ret_depth=False, scale_factor=None):
    """
    Parameters:
    - anchors: dictionary containing pixel coordinates and other metadata for each image
    - K: Intrinsic matrix for cameras
    - cam2w: Transformation matrix from camera to world coordinates
    - depthmaps: Dictionary of depth maps for each image
    - masks: Dictionary of binary masks for each image (same size as depthmaps)
    - base_focals: Optional focal length compensation
    - ret_depth: Whether to return depth values
    
    Returns:
    - pts3d_floor: List of 3D points belonging to the floor
    - pts3d_object: List of 3D points belonging to the object of interest
    - depth_out (optional): Depth values before world transformation
    """
    
    focals = K[:, 0, 0]
    invK = inv(K)
    all_pts3d = []
    all_pts3d_object = []
    all_pts3d_corners = []
    
    depth_out = []
    depth_out_object = []
    depth_out_corners = []

    
    for img, (pixels, idxs, offsets) in anchors.items():
        # Extract the mask for the current image
        mask = masks[img]
        if mask is None:
            # If mask is None, create a mask of ones matching the resolution of the depth map
            # Derive the size of the mask from the pixels tensor
            width = int(pixels[:, 0].max() + 1)  # Max x-coordinate + 1
            height = int(pixels[:, 1].max() + 1)  # Max y-coordinate + 1

            # Create a 2D mask of zeros with the derived size
            mask = np.zeros((height, width), dtype=np.uint8)
            print(f"Mask for image {img} is None. Using a mask of zeros.")

        mask_flat = mask.flatten()
        # From depthmaps to 3D points
        if base_focals is not None:
            if scale_factor is not None:
                offsets = (1 + (offsets - 1) * (base_focals[img] / focals[img]))*scale_factor[img]
            else:
                offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])
        # Ensure the mask is flattened to match 3D points
        
        # Project all points into 3D
        pts3d = proj3d(invK[img], pixels, depthmaps[img][idxs] * offsets)
        if ret_depth:
            depth_out.append(pts3d[..., 2])  # Save depth before rotation

        
        # Rotate to world coordinates
        pts3d = geotrf(cam2w[img], pts3d)
        all_pts3d.append(pts3d)
        
        if mask is not None:
            unique_masks = np.unique(mask_flat)  # Find unique values in mask_flat
            pts3d_object = []
            for mask_value in unique_masks:
                if mask_value == 0:
                    continue
                # Get the points corresponding to the current object (mask_value)
                object_mask = (mask_flat == mask_value)
                num_ones = np.count_nonzero(object_mask)
                # Separate points based on the mask value
                pts3d_object.append(pts3d[object_mask])  # Points where mask == mask_value (object)
                if ret_depth:
                    depth_out_object.append(pts3d[..., 2][object_mask])  # Depth for the current object

            all_pts3d_object.append(pts3d_object)

    if anchors_corners is not None and anchors_corners:
        for img, (pixels, idxs, offsets) in anchors_corners.items():
            
            # from depthmaps to 3d points
            if base_focals is None:
                pass
            else:
                # compensate for focal
                # depth + depth * (offset - 1) * base_focal / focal
                # = depth * (1 + (offset - 1) * (base_focal / focal))
                if scale_factor is None:
                    offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])
                else:
                    offsets = (1 + (offsets - 1) * (base_focals[img] / focals[img]))*scale_factor[img]
                    
            pts3d = proj3d(invK[img], pixels, depthmaps[img][idxs] * offsets)
            if ret_depth:
                depth_out_corners.append(pts3d[..., 2])  # before camera rotation

            # rotate to world coordinate
            pts3d = geotrf(cam2w[img], pts3d)
            all_pts3d_corners.append(pts3d)
        
        if len (all_pts3d_corners) > 0:
            stats = checkerboard_square_stats(all_pts3d_corners, pattern)

            print("Checkerboard square statistics:")
            for k, v in stats.items():
                print(f"{k}: {v:.4f}")

    if ret_depth:
        return all_pts3d, all_pts3d_object, depth_out, depth_out_object
    return all_pts3d, all_pts3d_object


def make_dense_pts3d(intrinsics, cam2w, depthmaps, canonical_paths, subsample, device='cuda'):
    base_focals = []
    anchors = {}
    confs = []
    for i, canon_path in enumerate(canonical_paths):
        (canon, canon2, conf), focal = torch.load(canon_path, map_location=device)
        confs.append(conf)
        base_focals.append(focal)
        H, W = conf.shape
        pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
        idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
        anchors[i] = (pixels, idxs[i], offsets[i])

    # densify sparse depthmaps
    pts3d, depthmaps_out = make_pts3d(anchors, intrinsics, cam2w, [
                                      d.ravel() for d in depthmaps], base_focals=base_focals, ret_depth=True)

    return pts3d, depthmaps_out, confs


@torch.no_grad()
def forward_mast3r(pairs, model, cache_path, desc_conf='desc_conf',
                   device='cuda', subsample=8, **matching_kw):
    res_paths = {}

    for img1, img2 in tqdm(pairs):
        start_time = time.time()
        idx1 = hash_md5(img1['instance'])
        idx2 = hash_md5(img2['instance'])

        path1 = cache_path + f'/forward/{idx1}/{idx2}.pth'
        path2 = cache_path + f'/forward/{idx2}/{idx1}.pth'
        path_corres = cache_path + f'/corres_conf={desc_conf}_{subsample=}/{idx1}-{idx2}.pth'
        path_corres2 = cache_path + f'/corres_conf={desc_conf}_{subsample=}/{idx2}-{idx1}.pth'

        if os.path.isfile(path_corres2) and not os.path.isfile(path_corres):
            score, (xy1, xy2, confs) = torch.load(path_corres2)
            torch.save((score, (xy2, xy1, confs)), path_corres)

        if not all(os.path.isfile(p) for p in (path1, path2, path_corres)):
            if model is None:
                continue
            
            res = symmetric_inference(model, img1, img2, device=device)
            X11, X21, X22, X12 = [r['pts3d'][0] for r in res]
            C11, C21, C22, C12 = [r['conf'][0] for r in res]
            descs = [r['desc'][0] for r in res]
            qonfs = [r[desc_conf][0] for r in res]
            # save
            torch.save(to_cpu((X11, C11, X21, C21)), mkdir_for(path1))
            torch.save(to_cpu((X22, C22, X12, C12)), mkdir_for(path2))

            # perform reciprocal matching
            corres = extract_correspondences(descs, qonfs, device=device, subsample=subsample)
            conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
            matching_score = (float(conf_score), float(corres[2].sum()), len(corres[2]))
            if cache_path is not None:
                torch.save((matching_score, corres), mkdir_for(path_corres))

        res_paths[img1['instance'], img2['instance']] = (path1, path2), path_corres
        # print("Forward MASt3R: ", time.time() - start_time)
    del model
    torch.cuda.empty_cache()

    return res_paths, cache_path


def symmetric_inference(model, img1, img2, device):
    shape1 = torch.from_numpy(img1['true_shape']).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2['true_shape']).to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return (res11, res21, res22, res12)


def extract_correspondences(feats, qonfs, subsample=8, device=None, ptmap_key='pred_desc'):
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs
    assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
    assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape

    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    qonf1 = []
    qonf2 = []
    # TODO add non symmetric / pixel_tol options
    for A, B, QA, QB in [(feat11, feat21, qonf11.cpu(), qonf21.cpu()),
                         (feat12, feat22, qonf12.cpu(), qonf22.cpu())]:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)

        idx1.append(np.r_[nn1to2[0], nn2to1[1]])
        idx2.append(np.r_[nn1to2[1], nn2to1[0]])
        qonf1.append(QA.ravel()[idx1[-1]])
        qonf2.append(QB.ravel()[idx2[-1]])
    # merge corres from opposite pairs
    H1, W1 = feat11.shape[:2]
    H2, W2 = feat22.shape[:2]
    cat = np.concatenate

    xy1, xy2, idx = merge_corres(cat(idx1), cat(idx2), (H1, W1), (H2, W2), ret_xy=True, ret_index=True)
    corres = (xy1.copy(), xy2.copy(), np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx]))

    return todevice(corres, device)


@torch.no_grad()
def prepare_canonical_data(imgs, tmp_pairs, subsample, order_imgs=False, min_conf_thr=0,
                           cache_path=None, device='cuda', camera_num=None, intrinsic_params=None, opt_process=None, **kw):
    canonical_views = {}
    pairwise_scores = torch.zeros((len(imgs), len(imgs)), device=device)
    canonical_paths = []
    preds_21 = {}
    if opt_process==PROCESS_ALL_IMAGES:
        folder_imgs = reshape_list(imgs, camera_num)
    else:
        folder_imgs = [imgs]
        if intrinsic_params is not None:
            intrinsic_params = [intrinsic_params]

    # print("Folder images length: ", len(folder_imgs))
    # print("Folder images: ", folder_imgs)

    for img in tqdm(imgs):
        found = False
        selected_list = None
        for i, folder_list in enumerate(folder_imgs):  
            if img in folder_list:
                selected_list = i
                found = True
                break  # Stop searching once found
        if not found:
            print(f"Image {img} not found in any folder list!")

        if cache_path:
            cache = os.path.join(cache_path, 'canon_views', hash_md5(img) + f'_{subsample=}_{kw=}.pth')
            canonical_paths.append(cache)
        try:
            (canon, canon2, cconf), focal = torch.load(cache, map_location=device)
        except IOError:
            # cache does not exist yet, we create it!
            canon = focal = None
        
        # collect all pred1
        n_pairs = sum((img in pair) for pair in tmp_pairs)

        ptmaps11 = None
        pixels = {}
        n = 0
        for (img1, img2), ((path1, path2), path_corres) in tmp_pairs.items():
            score = None
            if img == img1:
                X, C, X2, C2 = torch.load(path1, map_location=device)
                score, (xy1, xy2, confs) = load_corres(path_corres, device, min_conf_thr)
                pixels[img2] = xy1, confs
                if img not in preds_21:
                    preds_21[img] = {}
                # Subsample preds_21
                preds_21[img][img2] = X2[::subsample, ::subsample].reshape(-1, 3), C2[::subsample, ::subsample].ravel()

            if img == img2:
                X, C, X2, C2 = torch.load(path2, map_location=device)
                score, (xy1, xy2, confs) = load_corres(path_corres, device, min_conf_thr)
                pixels[img1] = xy2, confs
                if img not in preds_21:
                    preds_21[img] = {}
                preds_21[img][img1] = X2[::subsample, ::subsample].reshape(-1, 3), C2[::subsample, ::subsample].ravel()

            if score is not None:
                i, j = imgs.index(img1), imgs.index(img2)
                score = score[2]
                pairwise_scores[i, j] = score
                pairwise_scores[j, i] = score

                if canon is not None:
                    continue
                if ptmaps11 is None:
                    H, W = C.shape
                    ptmaps11 = torch.empty((n_pairs, H, W, 3), device=device)
                    confs11 = torch.empty((n_pairs, H, W), device=device)

                ptmaps11[n] = X
                confs11[n] = C
                n += 1
        if canon is None:
            canon, canon2, cconf = canonical_view(ptmaps11, confs11, subsample, **kw)
            del ptmaps11
            del confs11

        # compute focals
        H, W = canon.shape[:2]
        if intrinsic_params is None or intrinsic_params[selected_list] is None:
            pp = torch.tensor([W / 2, H / 2], device=device)
            if focal is None:
                focal = estimate_focal_knowing_depth(canon[None], pp, focal_mode='weiszfeld', min_focal=0.5, max_focal=3.5)
                if cache:
                    torch.save(to_cpu(((canon, canon2, cconf), focal)), mkdir_for(cache))
        else:
            focal = torch.tensor([intrinsic_params[selected_list]['focal']], device=device)
            pp = torch.tensor(intrinsic_params[selected_list]['pp'], device=device)

            if cache:
                torch.save(to_cpu(((canon, canon2, cconf), focal)), mkdir_for(cache))
        

        # extract depth offsets with correspondences
        core_depth = canon[subsample // 2::subsample, subsample // 2::subsample, 2]
        idxs, offsets = anchor_depth_offsets(canon2, pixels, subsample=subsample)

        canonical_views[img] = (pp, (H, W), focal.view(1), core_depth, pixels, idxs, offsets)

    return tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21

def load_corres(path_corres, device, min_conf_thr):
    score, (xy1, xy2, confs) = torch.load(path_corres, map_location=device)
    valid = confs > min_conf_thr if min_conf_thr else slice(None)
    # valid = (xy1 > 0).all(dim=1) & (xy2 > 0).all(dim=1) & (xy1 < 512).all(dim=1) & (xy2 < 512).all(dim=1)
    # print(f'keeping {valid.sum()} / {len(valid)} correspondences')
    return score, (xy1[valid], xy2[valid], confs[valid])


PairOfSlices = namedtuple(
    'ImgPair', 'img1, slice1, pix1, anchor_idxs1, img2, slice2, pix2, anchor_idxs2, confs, confs_sum')


def condense_data(imgs, tmp_paths, canonical_views, preds_21, dtype=torch.float32):
    # aggregate all data properly
    set_imgs = set(imgs)

    principal_points = []
    shapes = []
    focals = []
    core_depth = []
    img_anchors = {}
    tmp_pixels = {}

    for idx1, img1 in enumerate(imgs):
        # load stuff
        pp, shape, focal, anchors, pixels_confs, idxs, offsets = canonical_views[img1]

        principal_points.append(pp)
        shapes.append(shape)
        focals.append(focal)
        core_depth.append(anchors)

        img_uv1 = []
        img_idxs = []
        img_offs = []
        cur_n = [0]

        for img2, (pixels, match_confs) in pixels_confs.items():
            if img2 not in set_imgs:
                continue
            assert len(pixels) == len(idxs[img2]) == len(offsets[img2])
            img_uv1.append(torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=-1))
            img_idxs.append(idxs[img2])
            img_offs.append(offsets[img2])
            cur_n.append(cur_n[-1] + len(pixels))
            # store the position of 3d points
            tmp_pixels[img1, img2] = pixels.to(dtype), match_confs.to(dtype), slice(*cur_n[-2:])
        img_anchors[idx1] = (torch.cat(img_uv1), torch.cat(img_idxs), torch.cat(img_offs))

    all_confs = []
    imgs_slices = []
    corres2d = {img: [] for img in range(len(imgs))}

    for img1, img2 in tmp_paths:
        try:
            pix1, confs1, slice1 = tmp_pixels[img1, img2]
            pix2, confs2, slice2 = tmp_pixels[img2, img1]
        except KeyError:
            continue
        img1 = imgs.index(img1)
        img2 = imgs.index(img2)
        confs = (confs1 * confs2).sqrt()

        # prepare for loss_3d
        all_confs.append(confs)
        anchor_idxs1 = canonical_views[imgs[img1]][5][imgs[img2]]
        anchor_idxs2 = canonical_views[imgs[img2]][5][imgs[img1]]
        imgs_slices.append(PairOfSlices(img1, slice1, pix1, anchor_idxs1,
                                        img2, slice2, pix2, anchor_idxs2,
                                        confs, float(confs.sum())))

        # prepare for loss_2d
        corres2d[img1].append((pix1, confs, img2, slice2))
        corres2d[img2].append((pix2, confs, img1, slice1))

    all_confs = torch.cat(all_confs)
    corres = (all_confs, float(all_confs.sum()), imgs_slices)

    def aggreg_matches(img1, list_matches):
        pix1, confs, img2, slice2 = zip(*list_matches)
        all_pix1 = torch.cat(pix1).to(dtype)
        all_confs = torch.cat(confs).to(dtype)
        return img1, all_pix1, all_confs, float(all_confs.sum()), [(j, sl2) for j, sl2 in zip(img2, slice2)]
    corres2d = [aggreg_matches(img, m) for img, m in corres2d.items()]

    imsizes = torch.tensor([(W, H) for H, W in shapes], device=pp.device)  # (W,H)
    principal_points = torch.stack(principal_points)
    focals = [torch.tensor(focal).unsqueeze(0).to(dtype) if focal.ndim == 0 else focal for focal in focals]
    focals = torch.cat(focals)
    #focals = torch.stack(focals)

    # Subsample preds_21
    subsamp_preds_21 = {}
    for imk, imv in preds_21.items():
        subsamp_preds_21[imk] = {}
        for im2k, (pred, conf) in preds_21[imk].items():
            idxs = img_anchors[imgs.index(im2k)][1]
            subsamp_preds_21[imk][im2k] = (pred[idxs], conf[idxs])  # anchors subsample

    return imsizes, principal_points, focals, core_depth, img_anchors, corres, corres2d, subsamp_preds_21


def canonical_view(ptmaps11, confs11, subsample, mode='avg-angle'):
    assert len(ptmaps11) == len(confs11) > 0, 'not a single view1 for img={i}'

    # canonical pointmap is just a weighted average
    confs11 = confs11.unsqueeze(-1) - 0.999
    canon = (confs11 * ptmaps11).sum(0) / confs11.sum(0)

    canon_depth = ptmaps11[..., 2].unsqueeze(1)
    S = slice(subsample // 2, None, subsample)
    center_depth = canon_depth[:, :, S, S]
    center_depth = torch.clip(center_depth, min=torch.finfo(center_depth.dtype).eps)

    stacked_depth = F.pixel_unshuffle(canon_depth, subsample)
    stacked_confs = F.pixel_unshuffle(confs11[:, None, :, :, 0], subsample)

    if mode == 'avg-reldepth':
        rel_depth = stacked_depth / center_depth
        stacked_canon = (stacked_confs * rel_depth).sum(dim=0) / stacked_confs.sum(dim=0)
        canon2 = F.pixel_shuffle(stacked_canon.unsqueeze(0), subsample).squeeze()

    elif mode == 'avg-angle':
        xy = ptmaps11[..., 0:2].permute(0, 3, 1, 2)
        stacked_xy = F.pixel_unshuffle(xy, subsample)
        B, _, H, W = stacked_xy.shape
        stacked_radius = (stacked_xy.view(B, 2, -1, H, W) - xy[:, :, None, S, S]).norm(dim=1)
        stacked_radius.clip_(min=1e-8)

        stacked_angle = torch.arctan((stacked_depth - center_depth) / stacked_radius)
        avg_angle = (stacked_confs * stacked_angle).sum(dim=0) / stacked_confs.sum(dim=0)

        # back to depth
        stacked_depth = stacked_radius.mean(dim=0) * torch.tan(avg_angle)

        canon2 = F.pixel_shuffle((1 + stacked_depth / canon[S, S, 2]).unsqueeze(0), subsample).squeeze()
    else:
        raise ValueError(f'bad {mode=}')

    confs = (confs11.square().sum(dim=0) / confs11.sum(dim=0)).squeeze()
    return canon, canon2, confs


def anchor_depth_offsets(canon_depth, pixels, subsample=8):
    device = canon_depth.device

    # create a 2D grid of anchor 3D points
    H1, W1 = canon_depth.shape
    yx = np.mgrid[subsample // 2:H1:subsample, subsample // 2:W1:subsample]
    H2, W2 = yx.shape[1:]
    cy, cx = yx.reshape(2, -1)
    core_depth = canon_depth[cy, cx]
    assert (core_depth > 0).all()

    # slave 3d points (attached to core 3d points)
    core_idxs = {}  # core_idxs[img2] = {corr_idx:core_idx}
    core_offs = {}  # core_offs[img2] = {corr_idx:3d_offset}

    for img2, (xy1, _confs) in pixels.items():
        px, py = xy1.long().T

        # find nearest anchor == block quantization
        core_idx = (py // subsample) * W2 + (px // subsample)
        core_idxs[img2] = core_idx.to(device)

        # compute relative depth offsets w.r.t. anchors
        ref_z = core_depth[core_idx]
        pts_z = canon_depth[py, px]
        offset = pts_z / ref_z
        core_offs[img2] = offset.detach().to(device)

    return core_idxs, core_offs


def spectral_clustering(graph, k=None, normalized_cuts=False):
    graph.fill_diagonal_(0)

    # graph laplacian
    degrees = graph.sum(dim=-1)
    laplacian = torch.diag(degrees) - graph
    if normalized_cuts:
        i_inv = torch.diag(degrees.sqrt().reciprocal())
        laplacian = i_inv @ laplacian @ i_inv

    # compute eigenvectors!
    eigval, eigvec = torch.linalg.eigh(laplacian)
    return eigval[:k], eigvec[:, :k]


def sim_func(p1, p2, gamma):
    diff = (p1 - p2).norm(dim=-1)
    avg_depth = (p1[:, :, 2] + p2[:, :, 2])
    rel_distance = diff / avg_depth
    sim = torch.exp(-gamma * rel_distance.square())
    return sim


def backproj(K, depthmap, subsample):
    H, W = depthmap.shape
    uv = np.mgrid[subsample // 2:subsample * W:subsample, subsample // 2:subsample * H:subsample].T.reshape(H, W, 2)
    xyz = depthmap.unsqueeze(-1) * geotrf(inv(K), todevice(uv, K.device), ncol=3)
    return xyz


def spectral_projection_depth(K, depthmap, subsample, k=64, cache_path='',
                              normalized_cuts=True, gamma=7, min_norm=5):
    try:
        if cache_path:
            cache_path = cache_path + f'_{k=}_norm={normalized_cuts}_{gamma=}.pth'
        lora_proj = torch.load(cache_path, map_location=K.device)

    except IOError:
        # reconstruct 3d points in camera coordinates
        xyz = backproj(K, depthmap, subsample)

        # compute all distances
        xyz = xyz.reshape(-1, 3)
        graph = sim_func(xyz[:, None], xyz[None, :], gamma=gamma)
        _, lora_proj = spectral_clustering(graph, k, normalized_cuts=normalized_cuts)

        if cache_path:
            torch.save(lora_proj.cpu(), mkdir_for(cache_path))

    lora_proj, coeffs = lora_encode_normed(lora_proj, depthmap.ravel(), min_norm=min_norm)

    # depthmap ~= lora_proj @ coeffs
    return coeffs, lora_proj


def lora_encode_normed(lora_proj, x, min_norm, global_norm=False):
    # encode the pointmap
    coeffs = torch.linalg.pinv(lora_proj) @ x

    # rectify the norm of basis vector to be ~ equal
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    if global_norm:
        lora_proj *= coeffs[1:].norm() * min_norm / coeffs.shape[1]
    elif min_norm:
        lora_proj *= coeffs.norm(dim=1).clip(min=min_norm)
    # can have rounding errors here!
    coeffs = (torch.linalg.pinv(lora_proj.double()) @ x.double()).float()

    return lora_proj.detach(), coeffs.detach()


@torch.no_grad()
def spectral_projection_of_depthmaps(imgs, intrinsics, depthmaps, subsample, cache_path=None, **kw):
    # recover 3d points
    core_depth = []
    lora_proj = []

    for i, img in enumerate(tqdm(imgs)):
        cache = os.path.join(cache_path, 'lora_depth', hash_md5(img)) if cache_path else None
        depth, proj = spectral_projection_depth(intrinsics[i], depthmaps[i], subsample,
                                                cache_path=cache, **kw)
        core_depth.append(depth)
        lora_proj.append(proj)

    return core_depth, lora_proj

def apply_distortion(points, k1, k2, p1, p2, k3, K_fixed):
    """
    Applicate the distortion to the points
    """
    x, y = points[:, 0], points[:, 1]
    r2 = x**2 + y**2  # Distanza al quadrato dal centro
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    # Coordinate distorte
    x_distorted = x * radial + x_tangential
    y_distorted = y * radial + y_tangential
    x_dest = x_distorted * K_fixed[0,0] + K_fixed[0,2]
    y_dest = y_distorted * K_fixed[1,1] + K_fixed[1,2]
    return torch.stack([x_dest, y_dest], dim=1)

def reproj2d(Trf, pts3d):
    res = (pts3d @ Trf[:3, :3].transpose(-1, -2)) + Trf[:3, 3]
    clipped_z = res[:, 2:3].clip(min=1e-3)  # make sure we don't have nans!
    uv = res[:, 0:2] / clipped_z
    return uv.clip(min=-1000, max=2000)

def reproj2d_with_dist(Trf, pts3d, K_fixed, dist_coeffs_cam):
    res = (pts3d @ Trf[:3, :3].transpose(-1, -2)) + Trf[:3, 3]
    clipped_z = res[:, 2:3].clip(min=1e-3)  # make sure we don't have nans!
    uv = res[:, 0:2] / clipped_z
    k1 = dist_coeffs_cam[0]
    k2 = dist_coeffs_cam[1]
    p1 = dist_coeffs_cam[2]
    p2 = dist_coeffs_cam[3]
    k3 = dist_coeffs_cam[4]
    proj_points_distorted = apply_distortion(uv, k1, k2, p1, p2, k3, K_fixed)

    return proj_points_distorted


def bfs(tree, start_node):
    order, predecessors = sp.csgraph.breadth_first_order(tree, start_node, directed=False)
    ranks = np.arange(len(order))
    ranks[order] = ranks.copy()
    return ranks, predecessors


def compute_min_spanning_tree(pws):
    sparse_graph = sp.dok_array(pws.shape)
    for i, j in pws.nonzero().cpu().tolist():
        sparse_graph[i, j] = -float(pws[i, j])
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph)

    # now reorder the oriented edges, starting from the central point
    ranks1, _ = bfs(msp, 0)
    ranks2, _ = bfs(msp, ranks1.argmax())
    ranks1, _ = bfs(msp, ranks2.argmax())
    # this is the point farther from any leaf
    root = np.minimum(ranks1, ranks2).argmax()

    # find the ordered list of edges that describe the tree
    order, predecessors = sp.csgraph.breadth_first_order(msp, root, directed=False)
    order = order[1:]  # root not do not have a predecessor
    edges = [(predecessors[i], i) for i in order]

    return root, edges


def show_reconstruction(shapes_or_imgs, K, cam2w, pts3d, gt_cam2w=None, gt_K=None, cam_size=None, masks=None, **kw):
    viz = SceneViz()

    cc = cam2w[:, :3, 3]
    cs = cam_size or float(torch.cdist(cc, cc).fill_diagonal_(np.inf).min(dim=0).values.median())
    colors = 64 + np.random.randint(255 - 64, size=(len(cam2w), 3))

    if isinstance(shapes_or_imgs, np.ndarray) and shapes_or_imgs.ndim == 2:
        cam_kws = dict(imsizes=shapes_or_imgs[:, ::-1], cam_size=cs)
    else:
        imgs = shapes_or_imgs
        cam_kws = dict(images=imgs, cam_size=cs)
    if K is not None:
        viz.add_cameras(to_numpy(cam2w), to_numpy(K), colors=colors, **cam_kws)

    if gt_cam2w is not None:
        if gt_K is None:
            gt_K = K
        viz.add_cameras(to_numpy(gt_cam2w), to_numpy(gt_K), colors=colors, marker='o', **cam_kws)

    if pts3d is not None:
        for i, p in enumerate(pts3d):
            if not len(p):
                continue
            if masks is None:
                viz.add_pointcloud(to_numpy(p), color=tuple(colors[i].tolist()))
            else:
                viz.add_pointcloud(to_numpy(p), mask=masks[i], color=imgs[i])
    viz.show(**kw)
