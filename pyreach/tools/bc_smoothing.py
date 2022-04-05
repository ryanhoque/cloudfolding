"""
Code for getting LPAP actions from a trained model.
"""
import pickle
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from pyreach.tools.reach_keypoints.src.dataset import KeypointsDataset
from pyreach.tools.reach_keypoints.src.model import KeypointsGauss
from pyreach.tools.reach_keypoints.src.prediction import Prediction
from pyreach.tools.reach_keypoints.config import *
from pyreach.tools.basic_teleop import closest_pink_point
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels, get_boundary_points, get_boundary_points_erode, com_crop, undo_com_crop

DATADIR = 'data' # replace with directory where data/models are stored
WORKSPACE_CENTER = np.array((185, 345)) # (y, x) in pixel space
WORKSPACE_X_BOUNDS = (145, 530)
WORKSPACE_Y_BOUNDS = (40, 270) # 200

def analytic_second_point(image, first_point):
    """
    Calculate place point with AEP
    """
    closest_pp = closest_pink_point(image, *first_point, reverse_colors=True)
    vec1 = np.array(closest_pp) - first_point
    vec1 = ((vec1/np.linalg.norm(vec1)) * 70).astype(int)
    
    com = get_shirt_com(image)
    vec2 = (first_point - com) * 70 / np.linalg.norm(first_point - com)
    
    dest = first_point + (vec1 + vec2)/(3 if get_num_blue_pixels(image) < 35000 else 5) # if close to done, move lesss
    return dest

def clip(pt, y_bounds=WORKSPACE_Y_BOUNDS, x_bounds=WORKSPACE_X_BOUNDS):
    return np.array([max(min(y_bounds[1], pt[0]), y_bounds[0]), max(min(x_bounds[1], pt[1]), x_bounds[0])])

def get_some_point(heatmap, shirt_mask, orig_image, max_point=False, random=False):
    if not random:
        filtered_shirt = np.multiply(heatmap, shirt_mask)
    if random:
        points_that_work = []
        for i in range(shirt_mask.shape[0]):
            for j in range(shirt_mask.shape[1]):
                if shirt_mask[i, j] > 0.5:
                    points_that_work.append(np.array((i, j)))
        rand_point = points_that_work[np.random.randint(0, len(points_that_work))]
        return np.array([orig_image.shape[0]//2-160, orig_image.shape[1]//2-160]) + np.array(rand_point)
    elif max_point:
        print("max point instead")
        point = np.array(np.unravel_index(np.argmax(filtered_shirt), heatmap.shape))
        return point + np.array([orig_image.shape[0]//2-160, orig_image.shape[1]//2-160])
    else:
        # return a random point with a probability proportional to the value of the heatmap at that point
        heatmap_thresholded = np.where(filtered_shirt > 0.1, heatmap, 0)
        total_prob = 0
        for i in range(heatmap_thresholded.shape[0]):
            for j in range(heatmap_thresholded.shape[1]):
                total_prob += heatmap_thresholded[i, j]
        rand_prob = np.random.uniform(0, total_prob)
        curr_prob = 0
        for i in range(heatmap_thresholded.shape[0]):
            for j in range(heatmap_thresholded.shape[1]):
                curr_prob += heatmap_thresholded[i, j]
                if curr_prob > rand_prob:
                    ret_point = np.array([i, j]) + np.array([orig_image.shape[0]//2-160, orig_image.shape[1]//2-160])
                    print("Got a point at which value is:", heatmap_thresholded[i, j])
                    return ret_point

        # return max point otherwise 
        return get_some_point(heatmap, shirt_mask, orig_image, True)

buffer_len = 5
blue_pix_buffer = []

def do_nonmodel_stuff(image):
    """
    get_model_prediction() without actually calling the model (e.g. recentering, random move, etc)
    """
    global blue_pix_buffer
    bp = get_num_blue_pixels(image)
    print("NUM BLUE PIXELS", bp)
    blue_pix_buffer.append(bp)
    if len(blue_pix_buffer) > buffer_len:
        blue_pix_buffer = blue_pix_buffer[1:]

    if is_too_far(image, reverse_colors=False):
        # perform recentering move
        print('recentering...')
        pick_point = closest_blue_pixel_to_point(image, WORKSPACE_CENTER, reverse_colors=False, momentum=15)
        # Add noise
        pick_point = np.array(pick_point) + np.random.random(2) * 50 - 25
        pick_point = closest_blue_pixel_to_point(image, tuple(pick_point.tolist()), reverse_colors=False, momentum=15)
        
        place_point = pick_point + (WORKSPACE_CENTER - np.array(get_shirt_com(image, reverse_colors=False)))
        return {'pick': pick_point[::-1], 'place': place_point[::-1], 'done': False}

    # get random point if no progress is being made
    rnd_recovery_action = bp < 35000 and len(blue_pix_buffer) == buffer_len and np.mean(blue_pix_buffer[-buffer_len//2:]) < np.mean(blue_pix_buffer[:buffer_len//2])

    img = image[image.shape[0]//2-160:image.shape[0]//2+160, image.shape[1]//2-160:image.shape[1]//2+160, :]
    img = img.transpose(2, 0, 1)[::-1].copy() # RGB to BGR so the model likes it
    img_np = img.copy()
    shirt_mask = np.where(img_np[2,:,:] <= img_np[0,:,:],1,0)

    if (rnd_recovery_action):
        print("random recovery action...")
        blue_pix_buffer = []
        max_point = get_some_point(None, shirt_mask, image, random=rnd_recovery_action)
        second_point = analytic_second_point(image, max_point)

        # Clip both points to be within a reasonable area.
        max_point = clip(max_point)
        second_point = clip(second_point)

        max_point = tuple(map(lambda x: int(x), max_point.tolist()))
        second_point = tuple(map(lambda x: int(x), second_point.tolist()))
        max_point = closest_blue_pixel_to_point(image, max_point, reverse_colors=False, xbounds=WORKSPACE_X_BOUNDS, ybounds=WORKSPACE_Y_BOUNDS)
        return {'pick': max_point[::-1], 'place': second_point[::-1], 'done': get_num_blue_pixels(image) > 44500, 'coverage1': bp, 'coverage2': get_num_blue_pixels(image)} #41500
    else: # not random or recentering.
        return None

def get_folding_prediction(image, model_ckpt=DATADIR+"/fold_LPLP.pth"):
    """
    Analogue of get_model_prediction() below for LPLP folding
    """
    # model
    keypoints = KeypointsGauss(2, img_height=300, img_width=300)
    keypoints.load_state_dict(torch.load(model_ckpt))
    # cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        keypoints = keypoints.cuda()
    device = torch.device("cuda" if use_cuda else "cpu")
    prediction = Prediction(keypoints, 2, 300, 300, use_cuda)
    # preprocess image...
    orig_image = image.copy()
    image, crop_details = com_crop(image)
    image = torch.from_numpy(image.copy()).to(device).float()
    image = image.permute(2, 0, 1)

    heatmap = prediction.predict(image)
    heatmap = heatmap.detach().cpu().numpy()
    pick_y, pick_x = np.unravel_index(heatmap[0][0].argmax(), heatmap[0][0].shape)
    place_y, place_x = np.unravel_index(heatmap[0][1].argmax(), heatmap[0][1].shape)
    pick_x, pick_y, place_x, place_y = undo_com_crop(np.array([pick_x, pick_y, place_x, place_y]), crop_details)
    
    #pick_y, pick_x = clip(closest_blue_pixel_to_point(orig_image, (pick_y, pick_x), momentum=15))
    pick_y, pick_x = clip((pick_y, pick_x))
    place_y, place_x = clip((place_y, place_x))
    pick_x, pick_y = int(pick_x), int(pick_y)
    place_x, place_y = int(place_x), int(place_y)
    # viz
    #cv2.circle(orig_image, (pick_x, pick_y), 5, (255,0,0), -1)
    #cv2.circle(orig_image, (place_x, place_y), 5, (0,255,0), -1)
    #cv2.imshow('',orig_image)
    #cv2.waitKey()
    return {'pick': (pick_x, pick_y), 'place': (place_x, place_y)}

def get_model_prediction(image, model_ckpt=DATADIR+"/flatten_LPAP.pth", fully_analytic=False):
    """
    Main function for LPAP inference
    """
    global blue_pix_buffer
    bp = get_num_blue_pixels(image)
    print("NUM BLUE PIXELS", bp)
    blue_pix_buffer.append(bp)
    if len(blue_pix_buffer) > buffer_len:
        blue_pix_buffer = blue_pix_buffer[1:]

    if is_too_far(image, reverse_colors=False):
        print("Performing recentering move")
        # perform recentering move
        pick_point = closest_blue_pixel_to_point(image, WORKSPACE_CENTER, reverse_colors=False, momentum=15)
        # Add noise
        pick_point = np.array(pick_point) + np.random.random(2) * 25 - 12.5
        pick_point = closest_blue_pixel_to_point(image, tuple(pick_point.tolist()), reverse_colors=False, momentum=15)
        
        place_point = pick_point + (WORKSPACE_CENTER - np.array(get_shirt_com(image, reverse_colors=False)))

        # clip the pick and place points
        pick_point, place_point = clip(pick_point), clip(place_point)
        return {'pick': pick_point[::-1], 'place': place_point[::-1], 'done': False, 'type': 'recentering'}

    img = image[image.shape[0]//2-160:image.shape[0]//2+160, image.shape[1]//2-160:image.shape[1]//2+160, :]
    img = img.transpose(2, 0, 1)[::-1].copy() # RGB to BGR so the model likes it
    img_np = img.copy()

    # model
    keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    keypoints.load_state_dict(torch.load('%s'%model_ckpt))

    # cuda
    use_cuda = torch.cuda.is_available()
    use_cuda = True
    if use_cuda:
        torch.cuda.set_device(0)
        keypoints = keypoints.cuda()
    device = torch.device("cuda" if use_cuda else "cpu")

    prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)

    # Visualize the image to make sure it looks ok
    # plt.imshow(img.transpose(1, 2, 0))
    # plt.show()

    img = torch.from_numpy(img).to(device).float()
    heatmap = prediction.predict(img)
    heatmap = heatmap.detach().cpu().numpy().squeeze()

    shirt_mask = np.where(img_np[2, :, :] <= img_np[0, :, :], 1, 0)

    # get random point if no progress is being made
    rnd_recovery_action = bp < 38000 and len(blue_pix_buffer) == buffer_len and np.mean(blue_pix_buffer[-1:]) < np.mean(blue_pix_buffer[:buffer_len//2])
    max_point = get_some_point(heatmap, shirt_mask, image, random=fully_analytic or rnd_recovery_action)
    if (rnd_recovery_action):
        print("Performing random recovery action...")
        blue_pix_buffer = []
    second_point = analytic_second_point(image, max_point)

    # Clip both points to be within a reasonable area.
    max_point = clip(max_point)
    second_point = clip(second_point)

    max_point = tuple(map(lambda x: int(x), max_point.tolist()))
    second_point = tuple(map(lambda x: int(x), second_point.tolist()))
    # print(max_point, second_point)

    max_point = closest_blue_pixel_to_point(image, max_point, reverse_colors=False, xbounds=WORKSPACE_X_BOUNDS, ybounds=WORKSPACE_Y_BOUNDS, momentum=15)
    #second_point = closest_blue_pixel_to_point(image, second_point, reverse_colors=False)

    # Visualize if desired

    # plt.ion()
    # # overlay = cv2.addWeighted(img_np.transpose(1, 2, 0), 0.65, vis, 0.35, 0)
    # # overlay = cv2.circle(overlay, (pred_x,pred_y), 4, (0,0,0), -1)
    # plt.clf()
    img_show = img_np.transpose(1, 2, 0).copy()
    img_show[:, :, 0] = heatmap*255.0*2
    img_show[:, :, 1] = heatmap*255.0*2
    img_show[:, :, 2] = (0.7*img_show[:, :, 2]).astype(np.uint8)
    reshifted = max_point - np.array([image.shape[0]//2-160, image.shape[1]//2-160])

    return {'pick': max_point[::-1], 'place': second_point[::-1], 'done': get_num_blue_pixels(image) > 45000, 'coverage1': bp, 'coverage2': get_num_blue_pixels(image), 'type': 'recovery' if rnd_recovery_action else 'bc'} #41500
