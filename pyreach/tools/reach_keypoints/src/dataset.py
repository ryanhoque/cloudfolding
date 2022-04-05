import torch
import random
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
from datetime import datetime
import imgaug.augmenters as iaa

# No domain randomization
transform_none = iaa.Sequential([])

# Domain randomization
transform = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    # rot90
    iaa.Rot90([0, 1, 2, 3]),
    iaa.Rotate([-90, 90], mode='edge'),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        mode='edge'
    ),
], random_order=True)

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False):
    U.unsqueeze_(1).unsqueeze_(2)
    V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return normalize(G).double()
    return G.double()

def vis_gauss(img, gaussians):
    print(img.shape, gaussians.shape)
    img = img.cpu().numpy().transpose(1, 2, 0)[:, :, :3].squeeze().tolist()
    gaussians = gaussians.cpu().numpy().squeeze().tolist()
    h1 = np.array([gaussians[0], gaussians[0], gaussians[1]]).squeeze().transpose(1,2,0)
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    h1 = np.array(img).squeeze()
    output2 = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)
    cv2.imwrite('test2.png', output2)

class KeypointsDataset(Dataset):
    def __init__(self, img_folder, labels_folder, num_keypoints, img_height, img_width, transform, gauss_sigma=8, aug=True):
        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform if aug else transform_none

        self.imgs = []
        for filename in os.listdir(img_folder):
            self.imgs.append(os.path.join(img_folder, filename))

    def __getitem__(self, index):
        choice = random.randint(0, len(self.imgs) - 1)
        img_with_annot = self.imgs[choice]
        # np load it
        img, annot = np.load(img_with_annot, allow_pickle=True)
        # center crop img to 540 x 360
        img = torch.from_numpy(img).cuda().float()

        # check if annots is nested list
        if isinstance(annot[0], list):
            return self.__getitem__(index)

        #print(img.shape, annot, choice)

        gaussians_pick = gauss_2d_batch(img.shape[1], img.shape[0], self.gauss_sigma, torch.tensor([annot[0]]).cuda(), torch.tensor([annot[1]]).cuda()).float()
        gaussians_pick = gaussians_pick[:, img.shape[0]//2-160:img.shape[0]//2+160, img.shape[1]//2-160:img.shape[1]//2+160]
        gaussians_place = gauss_2d_batch(img.shape[1], img.shape[0], self.gauss_sigma, torch.tensor([annot[2]]).cuda(), torch.tensor([annot[3]]).cuda()).float()
        gaussians_place = gaussians_place[:, img.shape[0]//2-160:img.shape[0]//2+160, img.shape[1]//2-160:img.shape[1]//2+160]
        
        # concatenate along first axis
        gaussians = torch.cat((gaussians_pick, gaussians_place), 0)

        img = img[img.shape[0]//2-160:img.shape[0]//2+160, img.shape[1]//2-160:img.shape[1]//2+160, :]
        img = img.permute(2, 0, 1)
        #print(img.shape, gaussians.shape)

        # stack img and gaussians
        tmp = torch.cat((img, gaussians), dim=0).cpu().numpy().transpose(1, 2, 0)
        tmp = torch.from_numpy(self.transform(image=tmp).copy().transpose(2, 0, 1)).cuda().float()
        # extract img and gaussians
        img = tmp[:3, :, :]
        gaussians = tmp[3:, :, :]

        return img, gaussians
    
    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    NUM_KEYPOINTS = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    GAUSS_SIGMA = 10
    TEST_DIR = ""
    test_dataset = KeypointsDataset('/host/processed_data_folding/test',
                           '/host/processed_data_folding/test', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians = test_dataset[0]
    vis_gauss(img, gaussians)
 
