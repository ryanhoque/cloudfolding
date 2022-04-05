"""
Code for getting folding actions with ASM
"""
import pickle
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pyreach.tools.reach_keypoints.src.dataset import KeypointsDataset
from pyreach.tools.reach_keypoints.src.model import KeypointsGauss
from pyreach.tools.reach_keypoints.src.prediction import Prediction
from pyreach.tools.reach_keypoints.config import *
from pyreach.tools.basic_teleop import closest_pink_point
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels
from scipy.ndimage.interpolation import shift

DATADIR = 'data/'
WORKSPACE_CENTER = np.array((185, 345)) # (y, x) in pixel space
WORKSPACE_X_BOUNDS = (130, 540)
WORKSPACE_Y_BOUNDS = (40, 280) # 200

def get_keypoints_from_image(image):
    pass

def rotate(image, angle, center = None, scale = 1.0):
    image = image.astype(np.uint8)
    
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def match_image(img, target_mask):
    masked_img = img[:, :, 2] > img[:, :, 0]
    valid_region = np.zeros(masked_img.shape)
    valid_region[25:325, 150:550] = 1
    masked_img = masked_img * valid_region

    nonzero_indices = np.array(list(zip(*np.nonzero(masked_img))))
    img_center = np.mean(nonzero_indices, axis=0)
    nonzero_indices_target = np.array(list(zip(*np.nonzero(target_mask))))
    mask_center = np.mean(nonzero_indices_target, axis=0)
    shifted_target_mask = shift(target_mask, (int(img_center[0] - mask_center[0]), int(img_center[1] - mask_center[1])))

    best_score = -img.shape[0] * img.shape[1]
    best_angle, best_mask = None, None
    eyes = []
    scores = []
    for i in np.arange(0, 360, 1):
        rot_mask = rotate(shifted_target_mask, i, center=(img_center[1], img_center[0]))
        score = np.sum((np.logical_and(masked_img, rot_mask)))
        eyes.append(i)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_angle = i
            best_mask = rot_mask
    return img_center, mask_center, best_angle, best_mask, score

def get_relative_points(img, mask, points):
    center, mask_center, angle, best_mask, score = match_image(img, mask)

    relative_points = []
    # for each point in the input points, find the corresponding point in the best_mask
    for i in range(len(points)):
        relative_points.append(points[i] - mask_center)
    
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    rotated_points = []
    for point in relative_points:
        rotated_points.append(np.matmul(R, point))

    shifted_points = []
    for point in rotated_points:
        shifted_points.append(point + center)

    for p in shifted_points:
        plt.scatter(*p[::-1])

    return shifted_points

def get_folding_actions(image, action_file=DATADIR+'/shirt_script.npy'):  
    npy_file = np.load(action_file, allow_pickle=True).item()
    points = npy_file['points']
    for i in range(len(points)):
        points[i] = points[i][::-1]
    all_points = get_relative_points(image, npy_file['mask'], points)

    ret_pts = []
    for i in range(0, len(all_points), 2):
        ret_pts.append([all_points[i][::-1], all_points[i+1][::-1]])

    return ret_pts

def get_mask_score(image, action_file=DATADIR+'/shirt_script.npy'):
    npy_file = np.load(action_file, allow_pickle=True).item()
    center, mask_center, angle, best_mask, score = match_image(image, npy_file['mask'])
    print("MATCHING SCORE", score)
    return score > 37000
