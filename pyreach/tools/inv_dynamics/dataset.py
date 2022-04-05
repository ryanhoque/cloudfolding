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
from imgaug.augmentables import Keypoint, KeypointsOnImage
from pyreach.tools.analytic_benchmark import get_num_blue_pixels

DATADIR = 'data'

# No domain randomization
transform_none = iaa.Sequential([])

# Domain randomization
transform = iaa.Sequential([
    iaa.LinearContrast((0.85,1.15),per_channel=0.25),
    iaa.Add((-10,10), per_channel=True),
    iaa.GammaContrast((0.90, 1.10)),
    iaa.GaussianBlur(sigma=(0.0,0.3)),
    iaa.MultiplySaturation((0.95,1.05)),
    iaa.AdditiveGaussianNoise(scale=(0,0.0125*255)),
    #iaa.Fliplr(0.5),
    #iaa.Flipud(0.5),
    #iaa.Rot90([0, 1, 2, 3]),
    iaa.Rotate([-2, 2], mode='edge'),
    iaa.Affine(
        scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
        translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
        mode='edge'
    ),
], random_order=True)

class ForwardDynamicsDataset(Dataset): # CRL
    def __init__(self, img_folder, aug=True):
        self.transform = transform if aug else transform_none
        self.imgs = []
        for filename in os.listdir(img_folder):
            self.imgs.append(os.path.join(img_folder, filename))

    def __getitem__(self, index):
        img_file = self.imgs[index]
        # np load it
        img, act, delta_coverage = np.load(img_file, allow_pickle=True)
        h, w = img.shape[0], img.shape[1]
        assert h == 360 and w == 640
        # turn act to keypoints and perform data aug
        kps = [Keypoint(act[0], act[1]), Keypoint(act[2], act[3])]
        kps = KeypointsOnImage(kps, shape=img.shape)
        img, kps = self.transform(image=img, keypoints=kps)
        kps = kps.to_xy_array()
        # process img, kps: float & normalize, crop, HWC->CHW
        img = torch.from_numpy(img.copy()).float() / 255.
        img = img[:, 150:510, :]
        img = img.permute(2, 0, 1).cuda()
        # center crop kps and normalize
        kps[0][0] = max(min((kps[0][0] - 150) / 360., 1.), 0.)
        kps[0][1] = max(min((kps[0][1] - 0) / 360., 1.), 0.)
        kps[1][0] = max(min((kps[1][0] - 150) / 360., 1.), 0.)
        kps[1][1] = max(min((kps[1][1] - 0) / 360., 1.), 0.)
        kps = torch.tensor([kps[0][0], kps[0][1], kps[1][0], kps[1][1]])
        kps = kps.tile((250,))
        return img, kps.cuda(), torch.tensor(delta_coverage).cuda()

    def __len__(self):
        return len(self.imgs)

class InvDynamicsDataset(Dataset): # IDYN
    def __init__(self, img_folder, aug=True):
        self.transform = transform if aug else transform_none
        self.transform2 = transform.deepcopy() # for the second image

        self.imgs = []
        for filename in os.listdir(img_folder):
            self.imgs.append(os.path.join(img_folder, filename))

    def __getitem__(self, index):
        img_file = self.imgs[index]
        # np load it
        img, img2, act = np.load(img_file, allow_pickle=True)
        h, w = img.shape[0], img.shape[1]
        assert h == 360 and w == 640
        # turn act to keypoints and perform data aug
        kps = [Keypoint(act[0], act[1]), Keypoint(act[2], act[3])]
        kps = KeypointsOnImage(kps, shape=img.shape)
        img, kps = self.transform(image=img, keypoints=kps)
        kps = kps.to_xy_array()
        # do identical data aug on img2
        img2 = self.transform2(image=img2)
        # process img, img2, kps: float & normalize, crop, HWC->CHW
        img = torch.from_numpy(img.copy()).float() / 255.
        img = img[:, 150:510, :]
        img = img.permute(2, 0, 1)
        img2 = torch.from_numpy(img2.copy()).float() / 255.
        img2 = img2[:, 150:510, :]
        img2 = img2.permute(2, 0, 1)
        imgs = torch.stack([img, img2]).cuda()
        # center crop kps and normalize
        kps[0][0] = max(min((kps[0][0] - 150) / 360., 1.), 0.)
        kps[0][1] = max(min((kps[0][1] - 0) / 360., 1.), 0.)
        kps[1][0] = max(min((kps[1][0] - 150) / 360., 1.), 0.)
        kps[1][1] = max(min((kps[1][1] - 0) / 360., 1.), 0.)
        kps = torch.tensor([kps[0][0], kps[0][1], kps[1][0], kps[1][1]]).cuda()
        return imgs, kps
    
    def __len__(self):
        return len(self.imgs)

def format_data(input_dir=DATADIR+'/logs', output_dir=DATADIR+'/logs_proc'):
    cur_count = 0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir + '/train'):
        os.mkdir(output_dir + '/train')
        os.mkdir(output_dir + '/test')
    for i,f in enumerate(sorted(os.listdir(input_dir))):
        np_file = np.load(os.path.join(input_dir, f), allow_pickle=True)
        for ep in np_file:
            cimgs = ep["cimg"]
            actions = ep["act"]
            for j in range(len(actions)):
                if np.random.rand() > 0.1:
                    image_outpath = os.path.join(output_dir, 'train', '%05d.npy'%cur_count)
                else:
                    image_outpath = os.path.join(output_dir, 'test', '%05d.npy'%cur_count)
                res = (cimgs[j], cimgs[j+1], actions[j])
                np.save(image_outpath, res)
                cur_count += 1
    print('total samples', cur_count)

def format_forward_data(input_dir=DATADIR+'/logs_proc'):
    os.mkdir(input_dir+'/train-2')
    os.mkdir(input_dir+'/test-2')
    for filename in os.listdir(input_dir + '/train'):
        img_file = os.path.join(input_dir, 'train', filename)
        img, img2, act = np.load(img_file, allow_pickle=True)
        delta_coverage = (get_num_blue_pixels(img2) - get_num_blue_pixels(img)) / 50000.
        res = (img, act, delta_coverage)
        np.save(os.path.join(input_dir,'train-2',filename), res)
    for filename in os.listdir(input_dir + '/test'):
        img_file = os.path.join(input_dir, 'test', filename)
        img, img2, act = np.load(img_file, allow_pickle=True)
        delta_coverage = (get_num_blue_pixels(img2) - get_num_blue_pixels(img)) / 50000.
        res = (img, act, delta_coverage)
        np.save(os.path.join(input_dir,'test-2',filename),res)

if __name__ == '__main__':
    format_forward_data()
