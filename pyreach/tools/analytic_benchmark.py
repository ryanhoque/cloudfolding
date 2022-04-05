"""
Code with various analytical utilities.
"""
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

stride = 4
crop_size = 6
WORKSPACE_X_BOUNDS = (140, 530)
WORKSPACE_Y_BOUNDS = (30, 270)

def is_blue(rgb):
    return rgb[2] > rgb[0]

def is_boundary_region(crop):
    blues = []
    pinks = []
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            if is_blue(crop[i][j]):
                blues.append([i, j])
            else:
                pinks.append([i, j])      

    if len(blues) > 5 and len(pinks) > 5:
        return True, blues, pinks

    return False, blues, pinks

def closest_hot_point(image, edges, i, j):
    queue = [(i, j)]
    visited = np.ones(edges.shape) * -1
    visited[i, j] = 0
    pts_to_avg = []
    while len(queue) > 0:
        cur_point = queue[0]
        queue = queue[1:]
        if edges[cur_point[0], cur_point[1]] > 0:
            if visited[cur_point[0], cur_point[1]] < 80:
                pts_to_avg.append(cur_point)
            else:
                return None

        for n in [-1, 0, 1]:
            for m in [-1, 0, 1]:
                if visited[cur_point[0] - n, cur_point[1] - m] < 0:
                    visited[cur_point[0] - n, cur_point[1] - m] = visited[cur_point[0], cur_point[1]] + 1
                    queue.append((cur_point[0] - n, cur_point[1] - m))
    return None if not len(pts_to_avg) else np.mean(pts_to_avg, 0)

def get_boundary_points(image, edge_detect=False, close_pick=False):
    # iterate over image with 2d stride
    
    boundary_options = []
    for i in range(25, 325, stride):
        for j in range(150, 500, stride): #used to be 550 at upper limit
            boundary, blues, pinks = is_boundary_region(image[i:i + crop_size, j:j + crop_size])
            if boundary:
                pink_com = np.sum(np.array(pinks) + np.array([[i, j]] * len(pinks)), axis=0)/len(pinks)
                blue_com = np.sum(np.array(blues) + np.array([[i, j]] * len(blues)), axis=0)/len(blues)
                pink_com = pink_com.astype(int)
                blue_com = blue_com.astype(int)
                if close_pick:
                    pick = blue_com + 5*(pink_com - blue_com)
                else:
                    pick = blue_com + 4*(pink_com - blue_com)
                place = pink_com - 10*(pink_com - blue_com)
                boundary_options.append([pick, place])
    return boundary_options

def random_cloth_pick(image):
    shirt_mask = image[:,:,2] > image[:,:,0]
    # make shirt out of workspace 0
    shirt_mask[:,:WORKSPACE_X_BOUNDS[0]] = 0
    shirt_mask[:,WORKSPACE_X_BOUNDS[1]:] = 0
    shirt_mask[:WORKSPACE_Y_BOUNDS[0],:] = 0
    shirt_mask[WORKSPACE_Y_BOUNDS[1]:,:] = 0
    # get random point on shirt
    y, x = np.nonzero(shirt_mask)
    idx = np.random.randint(0, len(x))
    return (x[idx], y[idx])

def random_workspace_point(pick=None, max_dist=0.5):
    """
    if pick is not None, allow only up to max_dist away from pick point
    """
    max_x_dist = max_dist * (WORKSPACE_X_BOUNDS[1] - WORKSPACE_X_BOUNDS[0])
    max_y_dist = max_dist * (WORKSPACE_Y_BOUNDS[1] - WORKSPACE_Y_BOUNDS[0])
    x_valid, y_valid = False, False
    while not x_valid or not y_valid:
        x = np.random.randint(WORKSPACE_X_BOUNDS[0], WORKSPACE_X_BOUNDS[1])
        y = np.random.randint(WORKSPACE_Y_BOUNDS[0], WORKSPACE_Y_BOUNDS[1])
        x_valid = abs(x - pick[0]) < max_x_dist
        y_valid = abs(y - pick[1]) < max_y_dist
    return (x, y)

def get_boundary_points_erode(image):
    blue_mask = image[:, :, 0] > image[:, :, 2]
    blue_mask_eroded = cv2.erode(blue_mask.astype(np.uint8), np.ones((40, 40), dtype=np.uint8))
    # get diff between orig mask and eroded mask
    diff = np.logical_xor(blue_mask, blue_mask_eroded)
    diff[:50] = 0
    diff[325:] = 0
    diff[:, :150] = 0
    diff[:, 550:] = 0
    pts = np.argwhere(diff)
    # return a list of points
    return pts

def get_shirt_com(image, reverse_colors=False):
    stride_to_use = 1
    blue_points = []
    tot = 0
    for i in range(25, 325, stride_to_use):
        for j in range(150, 550, stride_to_use):
            tot += 1
            if reverse_colors != is_blue(image[i, j]):
                blue_points.append([i, j])
    return np.mean(blue_points, 0)

def com_crop(image, workspace_x=(125,520), workspace_y=(35,340), crop_width=300):
    # center crop around the visual COM but respect borders
    # return: cropped image & crop details
    y_com, x_com = get_shirt_com(image) 
    y, x = int(y_com), int(x_com)
    if x - crop_width // 2 < workspace_x[0]:
        x_lb = workspace_x[0]
        x_ub = workspace_x[0] + crop_width
    elif x + crop_width // 2 > workspace_x[1]:
        x_ub = workspace_x[1]
        x_lb = workspace_x[1] - crop_width
    else:
        x_lb = x - crop_width // 2
        x_ub = x + crop_width // 2
    if y - crop_width // 2 < workspace_y[0]:
        y_lb = workspace_y[0]
        y_ub = workspace_y[0] + crop_width
    else:
        y_lb = y - crop_width // 2
        y_ub = y + crop_width // 2
    image = image[y_lb:y_ub,x_lb:x_ub,:].copy()
    return image, (x_lb, x_ub, y_lb, y_ub)

def undo_com_crop(act, crop_details):
    x1, y1, x2, y2 = act
    x_lb, x_ub, y_lb, y_ub = crop_details
    y1, y2 = y1 + y_lb, y2 + y_lb
    x1, x2 = x1 + x_lb, x2 + x_lb
    return np.array([x1, y1, x2, y2])

def get_num_blue_pixels(image, reverse_colors=False):
    stride_to_use = 1
    blue_points = []
    tot = 0
    for i in range(25, 325, stride_to_use):
        for j in range(150, 550, stride_to_use):
            tot += 1
            if reverse_colors != is_blue(image[i, j]):
                blue_points.append([i, j])
    return len(blue_points)

def closest_blue_pixel_to_point(image, point, reverse_colors=False, strict=False, momentum=0, ybounds=(40, 290), xbounds=(130, 540)):
    closest_point, closest_dist = None, None
    for i in range(ybounds[0], ybounds[1], stride):
        for j in range(xbounds[0], xbounds[1], stride):
            if reverse_colors != is_blue(image[i, j]) and (not strict or image[i, j, 0] < 90):
                if (closest_dist is None or closest_dist > (i - point[0])**2 + (j - point[1])**2):
                    closest_point = np.array((i, j))
                    closest_dist = (i - point[0])**2 + (j - point[1])**2
    if momentum > 0 and any(closest_point != point):
        closest_pink_point = closest_blue_pixel_to_point(image, closest_point, reverse_colors=not reverse_colors, strict=False, momentum=0, ybounds=ybounds, xbounds=xbounds)
        return closest_point + momentum*(closest_point - closest_pink_point)/np.linalg.norm(closest_point - closest_pink_point)
    return closest_point

def is_too_far(image, reverse_colors=False):
    com = get_shirt_com(image, reverse_colors=reverse_colors)
    return com[0] < 90 or com[0] > 265 or com[1] < 250 or com[1] > 450

def get_boundary_pick_point(image, env, get_pixel_coords=False, new=False):
    image = image[...,::-1].copy()
    boundary_points = get_boundary_points(image, edge_detect=True, close_pick=True) if not new else get_boundary_points_erode(image)
    rand_index = np.random.randint(len(boundary_points))
    chosen = boundary_points[rand_index][0][::-1] if not new else boundary_points[rand_index][::-1]
    res = env.host.depth_camera.image().get_point_normal(chosen[0], chosen[1])
    pick = res[0]
    return (pick[0], pick[1]) if not get_pixel_coords else (pick[0], pick[1], chosen[0], chosen[1])

def get_analytic_pick_place_points(image, env, edge_detect=True):
    image = image[...,::-1].copy()
    boundary_points = get_boundary_points(image, edge_detect=True)
    rand_index = None #np.random.randint(len(boundary_points))
    best_rand_index = None
    best_height = -999999999999

    # Monte Carlo sample from boundary points and find one with the highest z coordinate
    for i in range(80):
        rand_index = np.random.randint(len(boundary_points))
        #print("boundary-points", boundary_points[rand_index])
        chosen = boundary_points[rand_index][0][::-1]
        res = env.host.depth_camera.image().get_point_normal(chosen[0], chosen[1])
        if res is None:
            print("None depth encountered")
            continue
        pick = res[0]
        if pick[2] > best_height:
            best_height = pick[2]
            best_rand_index = rand_index

    choice = boundary_points[best_rand_index] if random.random() < 0.7 else random.choice(boundary_points)
    #print(choice, rand_index, boundary_points[:4], best_height)

    # modify place point if using edge detection to improve direction of movement
    if edge_detect:
        edges = cv2.Canny(image=image, threshold1=75, threshold2=120)
        # filter out edges between blue/pink
        for i in range(25, 325, stride):
            for j in range(150, 550, stride):
                boundary, blues, pinks = is_boundary_region(image[i:i + crop_size, j:j + crop_size])
                if boundary:
                    edges[i- 5:i+crop_size+5, j-5:j+crop_size+5] = 0

        chp = closest_hot_point(image, edges, choice[0][0], choice[0][1])
        rando = random.random()
        choice[1] = choice[1]*rando + (choice[0] + (choice[0] - chp)/np.linalg.norm(choice[0] - chp) * 120)*(1 - rando) if chp is not None and any(choice[0] != chp) else choice[1]
        choice[0] = choice[0]*rando + chp*(1 - rando) if chp is not None else choice[0]
        choice[0] = choice[0].astype(int)
        choice[1] = choice[1].astype(int)

        if is_too_far(image):
            print("Moving shirt back to center")
            choice[1] = np.array([175, 350])

    for i in range(len(choice)):
        choice[i] = choice[i][::-1]

    window_name = "action preview"
    cv2.destroyAllWindows()
    cv2.namedWindow(window_name)
    cv2.circle(image, choice[0], 5, (255, 0, 0), 2)
    cv2.circle(image, choice[1], 5, (0, 255, 255), 2)
    if edge_detect and chp:
        cv2.circle(image, chp[::-1], 5, (0, 255, 0), 2)
    cv2.imshow(window_name, image)
    cv2.waitKey(2000)

    print("Choice: ", choice)

    return {'pick': (choice[0][0], choice[0][1]), 'place': (choice[1][0], choice[1][1])}
