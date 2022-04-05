import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/host/src')
from pyreach.tools.reach_keypoints.src.resnet_dilated import Resnet34_8s

class KeypointsGauss(nn.Module):
	def __init__(self, num_keypoints, img_height=480, img_width=640):
		super(KeypointsGauss, self).__init__()
		self.num_keypoints = num_keypoints
		self.num_outputs = self.num_keypoints
		self.img_height = img_height
		self.img_width = img_width
		self.resnet = Resnet34_8s()
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
		output = self.resnet(x) 
		heatmaps = self.sigmoid(output[:,:self.num_keypoints, :, :])
		return heatmaps

if __name__ == '__main__':
	model = KeypointsGauss(4).cuda()
	x = torch.rand((1,3,480,640)).cuda()
	result = model.forward(x)
	print(x.shape)
	print(result.shape)
