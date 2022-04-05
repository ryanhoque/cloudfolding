"""This script generates a GIF visualization from an experiment log."""
import sys
import moviepy.editor as mpy
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def is_blue(rgb):
    return rgb[2] > rgb[0]

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

def npy_to_gif(im_list, filename, fps=0.5):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')

for filename in os.listdir(sys.argv[1]):
    p = pickle.load(open(os.path.join(sys.argv[1], filename), 'rb'))
    images = p[0]['cimg']
    acts = p[0]['act']
    coverages = []

    for i in range(len(p[0]['act'])):
        coverages.append(get_num_blue_pixels(images[i]))
        cv2.circle(images[i], tuple([int(x) for x in acts[i][:2]]), 5, (255, 0, 0), -1)
        cv2.circle(images[i], tuple([int(x) for x in acts[i][2:]]), 5, (0, 255, 0), -1)
        # write text corresponding to number of blue pixels in image
        cv2.putText(images[i], str(coverages[-1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    npy_to_gif([im for im in images], filename)
    plt.clf()
    plt.plot(coverages)
    plt.savefig(filename + '.png')