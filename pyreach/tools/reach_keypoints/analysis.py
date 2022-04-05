import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np

model_ckpt = "model_2_1_24.pth"

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
keypoints.load_state_dict(torch.load('checkpoints/%s'%model_ckpt))

# cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = KeypointsDataset('/host/processed_data_folding/test',
    '/host/processed_data_folding/test', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA, aug=False)

for i in range(10):
    img_t, gt = test_dataset[i]
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    #print(img_t.shape, heatmap.shape)
    prediction.plot(img_t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8), heatmap, image_id=i)
 
