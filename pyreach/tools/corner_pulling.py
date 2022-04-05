"""
Code for the KP algorithm at inference time.
Most of the file is modeled after bc_smoothing.py.
"""
import pickle
import cv2
import os

from scipy.fftpack import shift
import torch
import numpy as np
import matplotlib.pyplot as plt

from pyreach.tools.reach_keypoints.src.dataset import KeypointsDataset
from pyreach.tools.reach_keypoints.src.model import KeypointsGauss
from pyreach.tools.reach_keypoints.src.prediction import Prediction
from pyreach.tools.reach_keypoints.config import *
from pyreach.tools.basic_teleop import closest_pink_point
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels, closest_blue_pixel_to_point

from pyreach.tools.select_point import SelectPoint

DATADIR = 'data' # replace with root directory where data/models are stored
WORKSPACE_CENTER = np.array((185, 345)) # (y, x) in pixel space
WORKSPACE_X_BOUNDS = (130, 540)
WORKSPACE_Y_BOUNDS = (40, 290)

def analytic_second_point(image, first_point):
    closest_pp = closest_pink_point(image, *first_point, reverse_colors=True)
    vec1 = np.array(closest_pp) - first_point
    vec1 = ((vec1/np.linalg.norm(vec1)) * 70).astype(int)
    
    com = get_shirt_com(image)
    vec2 = (first_point - com) * 70 / np.linalg.norm(first_point - com)
    
    dest = first_point + (vec1 + vec2)/(3 if get_num_blue_pixels(image) < 35000 else 5) # if close to done, move lesss
    return dest

def clip(pt, y_bounds=(50, 280), x_bounds=(130, 540)):
    return np.array([max(min(y_bounds[1], pt[0]), y_bounds[0]), max(min(x_bounds[1], pt[1]), x_bounds[0])])

def get_some_point(heatmap, shirt_mask, orig_image, max_point=True, random=False):
    if not random:
        filtered_shirt = np.multiply(heatmap, shirt_mask)
    if random:
        # choose random point on filtered shirt
        points_that_work = []
        for i in range(shirt_mask.shape[0]):
            for j in range(shirt_mask.shape[1]):
                if shirt_mask[i, j] > 0.5:
                    points_that_work.append(np.array((i, j)))
        return points_that_work[np.random.randint(0, len(points_that_work))]

    elif max_point:
        return np.array(np.unravel_index(np.argmax(filtered_shirt), heatmap.shape)) + np.array([orig_image.shape[0]//2-160, orig_image.shape[1]//2-160])
    else:
        # return a random point that meets the threshold
        points_that_work = []
        for i in range(filtered_shirt.shape[0]):
            for j in range(filtered_shirt.shape[1]):
                if filtered_shirt[i, j] > 0.3: # 0.5
                    points_that_work.append(np.array((i, j)))

        if not points_that_work:
            return get_some_point(heatmap, shirt_mask, orig_image, True)

        return points_that_work[np.random.randint(0, len(points_that_work))]   

buffer_len = 4
blue_pix_buffer = []

def do_nonmodel_stuff(image, force_random = False):
    """
    get_model_prediction without actually calling the model (eg recentering, random move, etc)
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

    if (rnd_recovery_action or force_random):
        print(force_random, rnd_recovery_action, bp, blue_pix_buffer, )
        print("random recovery action...")
        blue_pix_buffer = []
        max_point = get_some_point(None, shirt_mask, image, random=True)
        second_point = analytic_second_point(image, max_point)
        max_point = clip(max_point)
        second_point = clip(second_point)

        max_point = tuple(map(lambda x: int(x), max_point.tolist()))
        second_point = tuple(map(lambda x: int(x), second_point.tolist()))
        max_point = closest_blue_pixel_to_point(image, max_point, reverse_colors=False, xbounds=WORKSPACE_X_BOUNDS, ybounds=WORKSPACE_Y_BOUNDS)
        return {'pick': max_point[::-1], 'place': second_point[::-1], 'done': get_num_blue_pixels(image) > 44500, 'coverage1': bp, 'coverage2': get_num_blue_pixels(image)} #41500
    else: # not random or recentering.
        return None
with open(DATADIR+"/testkpdata.pkl", "rb") as f:
    shirt_data = pickle.load(f)
reference_tshirt = shirt_data[5]
reference_com = get_shirt_com(reference_tshirt[0])[::-1]
keypoint_model = torch.load(DATADIR+'/flatten-KP.pth')

kp_num_points = {"base":2, "sleeves":2,"collar":1}
from skimage.feature import peak_local_max

def generate_peak_points(heatmap, k, img, threshold=0.2, com = [0,0]):
    heatmap[heatmap < threshold] = 0
    
    new_com = cap_com(img, com)
    ydiff = new_com[0] -160
    xdiff =  new_com[1] -160
    peaks = peak_local_max(heatmap)
    if len(peaks) > 0:
        return np.flip((peaks[:min(peaks.shape[0], kp_num_points[k])] + [ydiff, xdiff]), axis = 1)
    else:
        return peaks
    

def cap_com(img, com):
    img_x = img.shape[1]
    img_y = img.shape[0]
    com[0] = int(com[0])
    com[1] = int(com[1])
    new_com = [int(max(min(com[0],img_x-1-160),160)), int(max(min(com[1],img_y-1-160),160))]
    print(com, new_com)
    return new_com[::-1]

def get_kp_heatmap(img, model=None, model_ckpt=None, com = None):
    assert model is not None or model_ckpt is not None
    new_com = cap_com(img, com)
    num_keypoints = 3
    if com is not None:
        img = img[new_com[0]-160:new_com[0]+160, new_com[1]-160:new_com[1]+160, :]
    else:  
        img = img[img.shape[0]//2-160:img.shape[0]//2+160, img.shape[1]//2-160:img.shape[1]//2+160, :]
    #plt.imshow(img)
    img = img.transpose(2, 0, 1)[::-1].copy() # RGB to BGR so the model likes it
    #plt.savefig("clip.png")
    # model
    keypoints = KeypointsGauss(num_keypoints, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    keypoints.load_state_dict(torch.load('%s'%model_ckpt) if model is None else model)

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
    return heatmap

selector = SelectPoint()
from datetime import datetime
def get_model_prediction(image, image_kps = None):
    ## call method to convert 3 type heatmaps to keypoints
    image_com =get_shirt_com(image)[::-1]
    if image_kps is None:
        image_kps = {k:generate_peak_points(heatmap, k, image, com = image_com) for heatmap,k in zip(get_kp_heatmap(image[...,::-1].copy(),model=keypoint_model, com = image_com),["base","sleeves","collar"])}
    obs = {k:np.array(v) for k,v in image_kps.items()}
    exp = {k:np.array(v) - np.array(reference_com) for k,v in reference_tshirt[1].items()}
    transf = find_min_transf(obs, exp, image_com).x
    
    shirt_mask = np.where(image[2,:,:] <= image[0,:,:],1,0)
    if sum([len(y) for y in image_kps.values()]) == 0:
        print('perf a random recovery to get keypoints')
        # blue_pix_buffer = []
        max_point = get_some_point(None, shirt_mask, image, random=True)
        second_point = get_some_point(image, max_point)

        # Clip both points to be within a reasonable area.
        max_point = clip(max_point)
        second_point = clip(second_point)

        max_point = tuple(map(lambda x: int(x), max_point.tolist()))
        second_point = tuple(map(lambda x: int(x), second_point.tolist()))
        max_point = closest_blue_pixel_to_point(image, max_point, reverse_colors=False, xbounds=WORKSPACE_X_BOUNDS, ybounds=WORKSPACE_Y_BOUNDS)
        return {'pick': max_point[::-1], 'place': second_point[::-1], 'done': get_num_blue_pixels(image) > 44500, 'coverage2': get_num_blue_pixels(image), 'kps':image_kps, 'force_reset':True} #41500
    
    ## rotate points to new locations
    transf_mat = rotMat(transf[0])
    rotated_pts = {k:(transf_mat@pt.T).T + image_com for k, pt in exp.items()}
    rotated_mat = np.vstack([x for x in rotated_pts.values()])
    
    point_bounds = [np.min(rotated_mat, axis = 0), np.max(rotated_mat, axis = 0)]
    point_conditional  =70 < point_bounds[0][0] < 600 and  70 < point_bounds[1][0] < 600\
             and  0< point_bounds[0][1] < 350 and  0< point_bounds[1][1] < 350
    print(f'cond: {point_conditional}, far: {is_too_far(image, reverse_colors=False)}')
    if sum([len(y) for y in image_kps.values()]) == 0 or is_too_far(image, reverse_colors=False) or not point_conditional:
        pick_point = closest_blue_pixel_to_point(image, WORKSPACE_CENTER, reverse_colors=False, momentum=15)
        pick_point = np.array(pick_point) + np.random.random(2) * 25 - 12.5
        pick_point = closest_blue_pixel_to_point(image, tuple(pick_point.tolist()), reverse_colors=False, momentum=15)
        place_point = pick_point + (WORKSPACE_CENTER - np.array(get_shirt_com(image, reverse_colors=False)))

        # clip the pick and place points
        pick_point, place_point = clip(pick_point), clip(place_point)

        return {'pick': pick_point[::-1], 'place': place_point[::-1], 'done': False, 'type': 'recentering', 'coverage2':get_num_blue_pixels(image),'kps':image_kps, 'force_reset':True}#,fig,ax
    
    ## find the points with the max delta from their respective positions
    error_matrix = {}
    for kp in rotated_pts:
        if len(obs[kp])  > 0:
            pts = obs[kp]
            error = sp.spatial.distance.cdist(rotated_pts[kp], pts, metric="euclidean")
            error_matrix[kp] = error
    max_err = []

    for kp in error_matrix:
        error = error_matrix[kp]
        if error.shape == (2,2):
            cost, smol = get_spatial_cost(error)
            if smol == 0:
                if error[0,0] > error[1,1]:
                    r,c = 0,0
                else:
                    r,c = 1,1
            else:
                if error[1,0] > error[0,1]:
                    r, c  = 0,1
                else:
                    r, c  = 1,0
        else:
            row_error = np.sum(np.min(error, axis = 1))
            col_error = np.sum(np.min(error, axis = 0))
            if row_error < col_error: ## use row error
                r = np.argmin(error, axis=1)
                values = [error[i,v] for i, v in enumerate(r)]
                c = np.argmax(values)
                if r.shape[0] > 1:
                    r = r[c]
                else:
                    r = r[0]
            else: ## use col error
                c = np.argmin(error, axis=0)
                values = [error[v,i] for i, v in enumerate(c)]
                r = np.argmax(values)
                if c.shape[0] > 1:
                    c = c[r]
                else:
                    c = c[0]
        if max_err == [] or max_err[1] < error[c, r]:
            max_err = [kp, error[c,r], (r,c)]
    start_pt = obs[max_err[0]][max_err[2][0]]
    end_pt = rotated_pts[max_err[0]][max_err[2][1]]
    ## Otherwise, return the pick-place action for the largest delta we need to cover
    return  {'pick':start_pt , 'place': end_pt, 'done': max_err[1] < 30, 'type': 'smoothen', 'coverage2':get_num_blue_pixels(image),'kps':image_kps} #fig, ax #41500


import scipy as sp

def rotMat(angle):
    return np.array([[np.cos(angle),-np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

def get_spatial_cost(spatial):
    assert len(spatial.shape) ==2
    if spatial.shape == (2,2):
        x = spatial[0,0]+spatial[1,1]
        y = spatial[1,0]+spatial[0,1]
        return min(x,y), np.argmin([x,y])
    else:
        return min(np.sum(np.min(spatial, axis = 1)), np.sum(np.min(spatial, axis = 0))), 9
    
def find_min_transf(obs, exp, com):
    '''
    obs are normal pts
    exp is zero centered pts
    '''
    def calculate_cost(rt):
        angle = rt[0]
        t = com
        mat = rotMat(angle)
        newExp = {k: mat@(np.array(v).T) for k,v in exp.items() if len(v) > 0}
        e = 0
        for kp in newExp:
            if kp in obs and len(obs[kp]) > 0:
                actual_pts = np.vstack([x for x in obs[kp]]) - t
                # print(actual_pts)
                pts = newExp[kp].T
                # print(new_pts)
                error = sp.spatial.distance.cdist(pts[:,:2], actual_pts, metric="euclidean")
                e += get_spatial_cost(error)[0]
        return e
    
    return sp.optimize.minimize(calculate_cost, [0])
    
import numpy as np
import copy
import scipy.ndimage as ndimage

import itertools

## get the center of mass of the t-shirt based on its image pixels
def get_shirt_com_naive(img):
    test = np.copy(img)
    
    ## optional: cut out the borders of the testbench
    test[0:10,:,:] = 0
    test[320:test.shape[0],:,:] = 0
    test[:,0:120,:] = 0
    test[:,550:600,:] = 0
    
    
    shirt_mask = test[:,:,2] > test[:,:,0]
    shirt_mask_2 = img[:,:,2] > img[:,:,0]
    return ndimage.center_of_mass(shirt_mask)[::-1], ndimage.center_of_mass(shirt_mask_2)[::-1]
# get_shirt_com(images[5][0])

def rigid_transform(A, B, A_ctr = None, B_ctr = None):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 2:
        raise Exception(f"matrix A is not 2xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 2:
        raise Exception(f"matrix B is not 2xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = A_ctr if A_ctr is not None else np.mean(A, axis=1)
    centroid_B = B_ctr if B_ctr is not None else np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # det(R) < R, reflection detected!, correcting for it ...
        Vt[1,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def calculate_optimal_transform(expected_kp, observed_kp, centroids = []):
    '''
    expected_kp: {'base':np.ndarray 2x2, 'sleeves': np.ndarray 2x2, 'collar': np.ndarray 2x1, 'com': np.ndarray 2x1}
    observed_kp: each ndarray is up to  {'base':np.ndarray 2x2, 'sleeves': np.ndarray 2x2, 'collar': np.ndarray 2x1, 'com': np.ndarray 2x1}
    '''
    
    ## Calculate all permutations of points for each type (to account for not having all keypoints necessary)
    run_permutes = {}
    for kp_type in expected_kp:
        permutes = itertools.combinations(list(range(len(expected_kp[kp_type]))), len(observed_kp[kp_type]))
        # print(list(permutes))
        run_permutes[kp_type] = [[list(expected_kp[kp_type][x]) for x in perm] for perm in permutes]
    ## Calculate the transforms for each point selection
    tfs = []
    for k,v in run_permutes.items():
        for pts in v:
            if len(pts) > 0:
                tfs.append(rigid_transform(np.vstack(list(pts)).reshape(2,-1), np.array(observed_kp[k]).reshape(2,-1), np.array(centroids[0]), np.array(centroids[1]) ))
    
    errors = []
    
    def transf_pt (transf, pts):
        transf_mat = np.vstack([np.hstack(transf), [0,0,1]])
        return (transf_mat@np.vstack([pts.T,np.ones((1,pts.shape[0]))])).T
    
   
    for tf in tfs:
        e = 0
        for kp in expected_kp:
            if len(observed_kp[kp]) > 0:
                actual_pts = pts = np.vstack([x for x in observed_kp[kp]])
                # print(actual_pts)
                pts = np.vstack([x for x in expected_kp[kp]])
                new_pts = transf_pt(tf, pts)
                # print(new_pts)
                error = sp.spatial.distance.cdist(new_pts[:,:2], actual_pts, metric="euclidean")
                e += min(np.sum(np.min(error, axis = 1)), np.sum(np.min(error, axis = 0)))
        errors.append(e)
    return tfs[np.argmin(errors)] if  len(errors) > 0 else []

def get_kp_deltas(kps, expected):
    deltas = {}
    expected = copy.deepcopy(expected)
    # taken_expected = {k:[] for k in expected}
    for kp_type in kps:
        deltas[kp_type] = []
        for point in kps[kp_type]:
            
            if kp_type in expected and len(expected[kp_type]) > 0:
                expected[kp_type] = np.array(expected[kp_type]).reshape(-1,2)
                base_point = np.array(point)
                dist_2 = np.sum((expected[kp_type] - base_point)**2, axis=1)
                min_idx = np.argmin(dist_2)
                # taken_expected[kp_type].append(min_idx)
                diff_pt = expected[kp_type][min_idx]
                expected[kp_type] = np.delete(expected[kp_type],min_idx, axis = 0)
                # print(expected[kp_type], diff_pt, base_point)
                deltas[kp_type].append(diff_pt - base_point)
            else:
                deltas[kp_type].append((np.inf, np.inf))
    return deltas

mat_1 = np.random.rand(2,5)
mat_2 = np.random.rand(2,4)
