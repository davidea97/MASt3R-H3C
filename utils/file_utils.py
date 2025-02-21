import os
import pandas as pd
import numpy as np
import cv2
import yaml
import torch 

def load_yaml_transformation(yaml_file):
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    matrix = fs.getNode("matrix").mat()
    fs.release()
    return matrix

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_scaled_heights(input_dir, scale):
    input_camera_height_dir = os.path.join(input_dir, "estimated_camera_heights")
    output_camera_height_dir = os.path.join(input_dir, "scaled_estimated_camera_heights")
    os.makedirs(output_camera_height_dir, exist_ok=True)

    yaml_files = sorted([f for f in os.listdir(input_camera_height_dir) if f.endswith(".yaml")])
    yaml_files = yaml_files[:-1]

    for filename in yaml_files:
        if filename.endswith(".yaml"):
            input_filepath = os.path.join(input_camera_height_dir, filename)
            output_filepath = os.path.join(output_camera_height_dir, filename)

            fs = cv2.FileStorage(input_filepath, cv2.FILE_STORAGE_READ)
            height = fs.getNode("height").real()
            fs.release()

            height_new = height * scale

            fs = cv2.FileStorage(output_filepath, cv2.FILE_STORAGE_WRITE)
            fs.write("height", height_new)
            fs.release()


def quaternion_distance(q1, q2):
    """
    Compute the distance between two quaternions (minimizing angular difference).
    Args:
        q1: Quaternion 1 (normalized, shape [4]).
        q2: Quaternion 2 (normalized, shape [4]).
    Returns:
        Angular distance in radians.
    """
    dot_product = torch.sum(q1 * q2, dim=-1)
    angular_distance = 2 * torch.acos(torch.clamp(torch.abs(dot_product), -1.0, 1.0))
    return angular_distance


def matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Args:
        R: 3x3 rotation matrix.
    Returns:
        Quaternion (shape [4]).
    """
    qw = torch.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
    return torch.stack([qw, qx, qy, qz])


def quaternion_to_matrix(q):
    """
    Converts a quaternion to a 3x3 rotation matrix.
    
    Args:
        q: Tensor of shape (4,), representing a quaternion [w, x, y, z].
        
    Returns:
        Tensor of shape (3, 3), representing the corresponding rotation matrix.
    """
    w, x, y, z = q

    # Compute the elements of the rotation matrix
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=0),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=0),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=0)
    ], dim=0)

    return R

def matrix_to_axis_angle(R):
    """
    Convert a rotation matrix to a 3-component axis-angle vector.
    """
    trace = torch.trace(R)
    angle = torch.acos(torch.clamp((trace - 1) / 2, min=-1.0, max=1.0))  # Clamp to valid range

    if angle < 1e-6:
        return torch.zeros(3, dtype=R.dtype, device=R.device, requires_grad=True)

    rx = (R[2, 1] - R[1, 2]) / (2 * torch.sin(angle) + 1e-8)  # Add epsilon to avoid division by zero
    ry = (R[0, 2] - R[2, 0]) / (2 * torch.sin(angle) + 1e-8)
    rz = (R[1, 0] - R[0, 1]) / (2 * torch.sin(angle) + 1e-8)

    return angle * torch.stack([rx, ry, rz])


def axis_angle_to_matrix(axis_angle):
    """
    Convert a 3-component axis-angle vector to a rotation matrix.
    """
    angle = torch.norm(axis_angle + 1e-8)  # Add a small epsilon to avoid division by zero
    axis = axis_angle / (angle + 1e-8)  # Normalize the axis safely

    # Skew-symmetric matrix for the axis
    x, y, z = axis.unbind()
    K = torch.stack([
        torch.tensor([0, -z, y]),
        torch.tensor([z, 0, -x]),
        torch.tensor([-y, x, 0])
    ], dim=0).to(axis.device)

    # Compute rotation matrix
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    R = torch.eye(3, device=axis.device) + sin_theta * K + (1 - cos_theta) * (K @ K)

    return R


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    Args:
        q1: First quaternion (shape [4]).
        q2: Second quaternion (shape [4]).
    Returns:
        Resultant quaternion (shape [4]).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    Args:
        q: Quaternion (shape [4]).
    Returns:
        Conjugate quaternion (shape [4]).
    """
    w, x, y, z = q
    return torch.tensor([w, -x, -y, -z])


def quaternion_rotate_vector(q, v):
    """
    Rotate a 3D vector using a quaternion.
    Args:
        q: Quaternion (shape [4]).
        v: Vector (shape [3]).
    Returns:
        Rotated vector (shape [3]).
    """
    v_quat = torch.tensor([0, *v])
    rotated_v = quaternion_multiply(quaternion_multiply(q, v_quat), quaternion_conjugate(q))
    return rotated_v[1:]  # Extract the vector part