import torch.nn as nn
import torchvision.models as models
from pyreach.tools.reach_keypoints.src.resnet import resnet34

class Resnet34_8s(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet34_8s, self).__init__()
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)
        self.resnet34_8s = resnet34_8s
        self._normal_initialization(self.resnet34_8s.fc)
        
    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = self.resnet34_8s(x)
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        return x
