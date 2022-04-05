"""
OpenCV GUI for human-specified pick-and-place actions.
"""
import cv2
import numpy as np

source, dest = None, None
image = None
analytic_second_point_glob = None
fold = False

def is_blue(rgb):
    return rgb[2] < rgb[0]

def click_callback(event, x, y, flags, param):
    global source, dest, image
    if event == cv2.EVENT_LBUTTONDOWN:
        source = (x, y)
        if analytic_second_point_glob:
            closest_pp = closest_pink_point(image, source[1], source[0])[::-1]
            vec = np.array(closest_pp) - np.array(source)
            #print(vec, closest_pp, source)
            vec = ((vec/np.linalg.norm(vec)) * 70).astype(int)
            dest = source + vec
            cv2.circle(image, source, 5, (255, 0, 0), -1)
            cv2.circle(image, dest, 5, (255, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP and not analytic_second_point_glob:
        dest = (x, y)
        cv2.arrowedLine(image, source, dest, (23, 126, 42), thickness=3)
        cv2.drawMarker(image, source, (23, 126, 42), thickness=3)

def closest_pink_point(image, i, j, reverse_colors=False):
    queue = [(i, j)]
    visited = np.ones(image.shape[:2]) * -1
    visited[i, j] = 0
    pts_to_avg = []
    while len(queue) > 0:
        cur_point = queue[0]
        queue = queue[1:]
        if reverse_colors == is_blue(image[cur_point[0], cur_point[1]]):
            if len(pts_to_avg) > 10:
                break
            pts_to_avg.append(cur_point)

        for n in [-1, 0, 1]:
            for m in [-1, 0, 1]:
                if visited[cur_point[0] - n, cur_point[1] - m] < 0:
                    visited[cur_point[0] - n, cur_point[1] - m] = visited[cur_point[0], cur_point[1]] + 1
                    queue.append((cur_point[0] - n, cur_point[1] - m))
    return np.mean(pts_to_avg, 0)

def get_human_pickplace_action(obs, analytic_second_point=False):
    """
    Given an image observation, allow user to provide a pick-and-place action. 
    Convert the pixels to workspace coordinates.
    analytic_second_point: if True, auto-compute place point with AEP
    """
    global source, dest, image, analytic_second_point_glob, fold
    analytic_second_point_glob = analytic_second_point
    fold = False

    obs = obs[...,::-1].copy() # bgr to rgb
    image = obs
    clone = obs.copy()
    window_name = "Click+Drag to Demo, 'c' Confirm, 'r' Redo:"
    cv2.destroyAllWindows()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_callback)
    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            source, dest = None, None
            image = clone.copy()
        elif key == ord("c"):
            break
        elif key == ord('f'):
            fold = True
            break
    assert source is not None and dest is not None, "No action specified."
    return {'pick': source, 'place': dest, 'fold': fold}
 