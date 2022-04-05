"""This script computes stats like max coverage and computation time 
from experiment logs.
"""
import sys
from pyreach.tools.analytic_benchmark import get_num_blue_pixels
import numpy as np
import pickle
import cv2

files = sys.argv[1:]
num_eps = 0
max_cov = []
num_acts = []
times = []
for file in files:
        p = pickle.load(open(file, 'rb'))
        for ep in p:
                num_eps += 1
                images = ep['cimg']
                maxcov = 0
                acts = 0
                for inf in ep['info']:
                    if inf['smoothing'] == 0:
                        break
                    acts += 1
                acts = min(acts, 100)
                num_acts.append(acts)
                for i in range(acts+1):
                    maxcov = max(maxcov, get_num_blue_pixels(images[i]))
                max_cov.append(maxcov)
                for i in range(acts):
                    times.append(ep['full_obs'][i+1]['server']['latest_ts'] - ep['full_obs'][i]['server']['latest_ts'])
print('num eps', num_eps)
print('max_cov', max_cov)
print('num_acts', num_acts)
print('times', np.array(times).mean(), np.array(times).std())
