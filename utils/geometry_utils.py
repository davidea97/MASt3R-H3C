import os
import numpy as np
from utils.file_utils import load_yaml_transformation

def accumulate_absolute_transformations(transformations):

    accumulated = [transformations[0]]
    for i in range(1, len(transformations)):
        accumulated.append(np.dot(accumulated[-1], transformations[i]))
    return accumulated

def accumulate_relative_transformations(transformations):

    accumulated = []
    for i in range(len(transformations)):
        accumulated.append(transformations[i])
    return accumulated

def load_transformations(gt_folder, est_folder, common_indices):

    gt_transformations = []
    est_transformations = []

    for index in sorted(common_indices):
        gt_file = os.path.join(gt_folder, f"{index:04}.yaml")
        est_file = os.path.join(est_folder, f"{index:04}.yaml")
        # print("------- Iteration: ", index)
        # print("GT file: ", gt_file)
        # print("Est file: ", est_file)

        # Carica le trasformazioni
        gt_matrix = load_yaml_transformation(gt_file)
        est_matrix = load_yaml_transformation(est_file)
        
        gt_transformations.append(gt_matrix)
        est_transformations.append(est_matrix)
    
    return gt_transformations, est_transformations

def calculate_weights(robot_rotations, robot_translations):
    num_poses = len(robot_rotations)
    rotation_diffs = np.zeros(num_poses)
    translation_diffs = np.zeros(num_poses)

    for i in range(num_poses):
        # Calcola la rotazione relativa tra pose consecutive
        angle = np.arccos((np.trace(robot_rotations[i]) - 1) / 2)
        rotation_diffs[i] = abs(angle)  # Usa il valore assoluto dell'angolo

        # Calcola la distanza tra traslazioni consecutive
        translation_diffs[i] = np.linalg.norm(robot_translations[i])

    # Assegna pesi inversamente proporzionali alla quantità di rotazione
    weights = 1 / (rotation_diffs + 1e-6)  # Evita la divisione per zero
    #weights /= np.sum(weights)  # Normalizza i pesi per sommare a 1

    return weights

"""def robust_umeyama_alignment(source_points, target_points):
    
    src_centroid = np.mean(source_points, axis=0)
    tgt_centroid = np.mean(target_points, axis=0)

    src_centered = source_points - src_centroid
    tgt_centered = target_points - tgt_centroid

    covariance_matrix = np.dot(tgt_centered.T, src_centered) / source_points.shape[0]

    U, S, Vt = np.linalg.svd(covariance_matrix)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    src_distances = np.linalg.norm(src_centered, axis=1)
    tgt_distances = np.linalg.norm(tgt_centered, axis=1)
    scale = np.median(tgt_distances / src_distances)  

    translation = tgt_centroid - scale * np.dot(R, src_centroid)

    return R, scale, translation"""

def robust_umeyama_alignment(cam_translations, rob_translations):
    """
    Performs a robust alignment of cam_translations to rob_translations using Umeyama method.
    
    Args:
        cam_translations (np.ndarray): Nx3 array of camera translations.
        rob_translations (np.ndarray): Nx3 array of robot translations.

    Returns:
        R (np.ndarray): 3x3 rotation matrix.
        scale (float): Scale factor.
        translation (np.ndarray): 3x1 translation vector.
    """
    assert cam_translations.shape == rob_translations.shape, "Input shapes must match."
    n = cam_translations.shape[0]

    # Calculate the means of both translation sets
    mean_cam = np.mean(cam_translations, axis=0)
    mean_rob = np.mean(rob_translations, axis=0)

    # Center the translations
    cam_centered = cam_translations - mean_cam
    rob_centered = rob_translations - mean_rob

    # Compute the covariance matrix
    covariance_matrix = np.dot(rob_centered.T, cam_centered) / n

    # SVD decomposition
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Correct reflection issue
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    # Compute scale
    #gt_distances = np.linalg.norm(np.diff(rob_centered, axis=0), axis=1)
    #est_distances_aligned = np.linalg.norm(np.diff(cam_centered, axis=0), axis=1)

    gt_distances = np.linalg.norm(rob_centered, axis=1)
    est_distances_aligned = np.linalg.norm(cam_centered, axis=1)
    #print("GT distances: ", gt_distances)
    #print("Est distances: ", est_distances_aligned)

    scale_factors = gt_distances / est_distances_aligned
    scale = np.mean(scale_factors)

    # Compute translation
    translation = mean_rob - scale * R @ mean_cam

    return R, scale, translation
    

def robust_weighted_umeyama_alignment(est_points, gt_points, weights=None):

    if weights is None:
        weights = np.ones(est_points.shape[0])

    # Normalizza i pesi
    weights = weights / np.sum(weights)

    # Calcola le medie pesate
    est_mean = np.average(est_points, axis=0, weights=weights)
    gt_mean = np.average(gt_points, axis=0, weights=weights)

    # Centra i punti intorno al baricentro pesato
    est_centered = est_points - est_mean
    gt_centered = gt_points - gt_mean

    # Applica i pesi ai punti centrati
    est_weighted = est_centered * weights[:, np.newaxis]
    gt_weighted = gt_centered * weights[:, np.newaxis]

    # Calcola la matrice di covarianza pesata
    covariance_matrix = np.dot(gt_weighted.T, est_centered)

    # Decomposizione SVD
    U, D, Vt = np.linalg.svd(covariance_matrix)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    # Calcola la matrice di rotazione R
    R = np.dot(U, np.dot(S, Vt))

    # Calcola il fattore di scala
    var_est = np.sum(weights * np.sum(est_centered ** 2, axis=1))
    scale = np.trace(np.dot(np.diag(D), S)) / var_est

    # Calcola la traslazione
    translation = gt_mean - scale * np.dot(est_mean, R)

    return R, scale, translation

def apply_transformation_to_pose(pose, R, t, s):
    # Estrai la parte rotazionale e traslazionale dalla posa
    rotation_part = pose[:3, :3]
    translation_part = pose[:3, 3]
    
    # Applica scala, rotazione e traslazione alla parte rotazionale e traslazionale
    new_rotation = s * np.dot(R, rotation_part)
    new_translation = s * np.dot(R, translation_part) + t

    # Costruisci la nuova matrice di trasformazione 4x4
    new_pose = np.eye(4)
    new_pose[:3, :3] = new_rotation
    new_pose[:3, 3] = new_translation
    
    return new_pose

def apply_translation_alignment(est_poses, aligned_translations):
    aligned_poses = []
    for i, pose in enumerate(est_poses):
        new_pose = pose.copy()
        new_pose[:3, 3] = aligned_translations[i]
        aligned_poses.append(new_pose)
    return aligned_poses
