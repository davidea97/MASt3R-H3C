import glob
import os
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

def generate_image_list(folder_path):
    """
    Generates a list of image paths organized by subfolders (e.g., camera1, camera2, etc.).

    Args:
        folder_path (str): The root folder containing subfolders named camera1, camera2, etc.

    Returns:
        list of lists: A list where each sublist contains paths to images from one camera folder.
    """
    # Get all subdirectories within the folder_path
    subfolders = sorted([
        os.path.join(folder_path, d) for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('camera')
    ])

    # Initialize the matrix (list of lists) to store images from each subfolder
    image_list = [[] for _ in range(len(subfolders))]

    for i, subfolder in enumerate(subfolders):
        # Find all .png files in the subfolder (adjust for other extensions if needed)
        image_paths = glob.glob(os.path.join(os.path.join(subfolder, "image"), "*.png"))

        # Normalize paths to use forward slashes for compatibility
        image_paths = [path.replace("\\", "/") for path in image_paths]

        # Sort the paths to ensure consistent ordering
        image_paths.sort()

        # Add the list of image paths from this subfolder to the matrix
        image_list[i] = image_paths

    return image_list, subfolders


def generate_mask_list(subfolders, image_list, image_ext=None):
    """
    Generates a list of mask image paths organized by subfolders (e.g., camera1, camera2, etc.),
    but only includes folders that contain a "mask" subfolder.

    Args:
        folder_path (str): The root folder containing subfolders named camera1, camera2, etc.

    Returns:
        list of lists: A list where each sublist contains paths to mask images from one camera folder.
    """
    # Get all subdirectories within the folder_path
    # subfolders = sorted([
    #     os.path.join(folder_path, d) for d in os.listdir(folder_path)
    #     if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('camera')
    # ])

    mask_list = [[] for _ in range(len(subfolders))]

    for i, subfolder in enumerate(subfolders):
        # Check if the "mask" subfolder exists
        mask_folder = os.path.join(subfolder, "masks")
        if not os.path.exists(mask_folder) or not os.path.isdir(mask_folder):
            print(f"Skipping: {mask_folder} (does not exist)")
            continue

        # Find all .png files in the "mask" subfolder (adjust for other extensions if needed)
        if image_ext is None:
            image_ext = ".png"
        mask_paths = glob.glob(os.path.join(mask_folder, f"*{image_ext}"))
        # Normalize paths to use forward slashes for compatibility
        #mask_paths = np.empty(len(image_list[i]))
        mask_paths = [path.replace("\\", "/") for path in mask_paths]
        # Sort the paths to ensure consistent ordering
        mask_paths.sort()
        mask_paths = mask_paths[:len(image_list[i])]

        mask_list[i] = mask_paths

    if len(mask_list)==1:
        mask_list = mask_list[0]
    return mask_list


def read_intrinsics(camera_folders, config, intrinsic_file="intrinsic_pars_file.yaml"):
    intrinsics = []
    dist_coeffs = []
    for camera_folder in camera_folders:
        with open(os.path.join(camera_folder, intrinsic_file), "r") as file:
            data = yaml.safe_load(file)

        target_image_size = config['image_size']
        img_width = data['img_width']
        img_height = data['img_height']
        original_size = max(img_width, img_height)
        scale_factor = target_image_size / original_size

        fx = data['fx']*scale_factor
        fy = data['fy']*scale_factor
        cx = data['cx']*scale_factor
        cy = data['cy']*scale_factor
        intrinsics.append({'focal': fx,       
                            'pp': (cx, cy)}
        )

        k0 = data['dist_k0']
        k1 = data['dist_k1']
        k2 = data['dist_k2']
        px = data['dist_px']
        py = data['dist_py']

        dist_coeffs.append(np.array([k0, k1, px, py, k2]))
    
    return intrinsics, dist_coeffs


def reshape_list(lst, N):
    k, m = divmod(len(lst), N)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(N)]


def rotation_matrix_to_rpy(R):
    """
    Convert a 3x3 rotation matrix to roll (X), pitch (Y), and yaw (Z) angles.
    Assumes ZYX intrinsic rotation (yaw → pitch → roll).
    Returns angles in degrees.
    """
    roll = np.arctan2(R[2,1], R[2,2])  # atan2(R32, R33)
    pitch = np.arcsin(-R[2,0])         # asin(-R31)
    yaw = np.arctan2(R[1,0], R[0,0])   # atan2(R21, R11)

    return roll, pitch, yaw  # Convert to degrees

def read_transformations(file_path):
    """Reads a TXT file containing transformation data (tx, ty, tz, rx, ry, rz)."""
    transformations = np.loadtxt(file_path, delimiter=",")
    return transformations


def compute_errors(gt, est, camera_num, camera_selected=0):
    """
    Computes translation and rotation errors between ground truth (gt) and estimated (est).
    
    - gt, est: Nx6 arrays where each row represents (tx, ty, tz, rx, ry, rz).
    """
    if camera_num>1:
        # Compute translation error (Euclidean distance)
        trans_error = np.linalg.norm(gt[:, :3] - est[:, :3], axis=1)

        # Compute rotation error using angle-axis representation
        rot_gt = R.from_rotvec(gt[:, 3:])  # Convert to rotation objects
        rot_est = R.from_rotvec(est[:, 3:])
    else:
        # Compute translation error (Euclidean distance)
        est = est[0]
        trans_error = np.linalg.norm(gt[:3] - est[:3], axis=0)

        # Compute rotation error using angle-axis representation
        rot_gt = R.from_rotvec(gt[3:])  # Convert to rotation objects
        rot_est = R.from_rotvec(est[3:])
    
    # Compute relative rotation (GT^-1 * EST)
    rot_rel = rot_gt.inv() * rot_est  # Relative rotation
    rot_error = rot_rel.magnitude()   # Extract the angle error in radians
    
    return trans_error, rot_error  # Convert rotation error to degrees
