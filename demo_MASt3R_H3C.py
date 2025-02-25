#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo executable
# --------------------------------------------------------
import os
import torch
import tempfile
from contextlib import nullcontext

from mast3r.demo_3D_scaled_representation import get_args_parser, main_demo

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from utils.file_utils import load_config
from mast3r.utils.general_utils import generate_image_list, generate_mask_list, read_intrinsics

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp
from utils.file_utils import *

from Grounded_SAM_2.sam2_mask_tracking import MaskGenerator 
import glob
import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()

    # parser.add_argument('--model_name', type=str, default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument('--input_folder', type=str, default="dust3r/croco/assets")
    parser.add_argument('--outdir', type=str, default="output")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument('--mask_floor', type=str2bool, default=True, help="True or False for floor mask generation")
    parser.add_argument('--subset_size', type=int, default=0, help="Number of images to use for the reconstruction")
    parser.add_argument('--use_intrinsics', type=str2bool, default=True, help="Use intrinsic parameters for the cameras")
    parser.add_argument('--calibrate_sensor', type=str2bool, default=True, help="Use robot motion to perform the calibration step")

    args = parser.parse_args()
    set_print_with_timestamp()

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)
    config = load_config(args.config)
    
    image_list, subfolders = generate_image_list(args.input_folder)
    robot_poses = [[] for _ in range(len(subfolders))]
    image_sublist = [[] for _ in range(len(subfolders))]
    if args.calibrate_sensor:
        print("Use robot motion to perform the calibration step")
        for i, subfolder in enumerate(subfolders):
            yaml_files = sorted(glob.glob(f"{subfolder}/relative_rob_poses/*.yaml"))
            for yaml_file in yaml_files:

                # Read matrix from YAML file using OpenCV
                fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
                matrix = fs.getNode("matrix").mat()

                # Scale translation of the matrix
                scale = 1.0
                scale = 0.84
                matrix[:3, 3] = matrix[:3, 3] * scale

                fs.release()
                robot_poses[i].append(torch.tensor(matrix))
    else:
        robot_poses = None

    if args.subset_size > 0:
        for i in range(len(image_list)):
            image_sublist[i] = image_list[i][:args.subset_size]
        if robot_poses is not None:
            for i in range(len(robot_poses)):
                robot_poses[i] = robot_poses[i][:args.subset_size-1]
    else:
        image_sublist = image_list

    if args.mask_floor:
        mask_generator = MaskGenerator(config, image_sublist, subfolders)
        print("Generating masks...")
        objects, image_ext = mask_generator.generate_masks()
    mask_list = generate_mask_list(args.input_folder, image_sublist, image_ext=image_ext)

    intrinsic_params_vec = []
    dist_coeffs = []
    if args.use_intrinsics:
        intrinsic_params_vec, dist_coeffs = read_intrinsics(subfolders, config)
        print("Intrinsics: ", intrinsic_params_vec)
        print("Distortion coefficients: ", dist_coeffs)
    else:
        for i in range(len(subfolders)):
            intrinsic_params_vec.append(None)
            dist_coeffs.append(None)
            print("Intrinsic parameters not provided. Estimating intrinsics..")

    def get_context(tmp_dir):
        return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None \
            else nullcontext(tmp_dir)
    with get_context(args.tmp_dir) as tmpdirname:
        cache_path = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)
        main_demo(cache_path, model, config, args.device, server_name, args.server_port, image_sublist, mask_list, args.silent,
                  intrinsic_params_vec, dist_coeffs, robot_poses, share=args.share, gradio_delete_cache=args.gradio_delete_cache)
