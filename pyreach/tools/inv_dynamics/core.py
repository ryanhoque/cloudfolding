from .resnet import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F

class InvDynNet(nn.Module):
    """IDYN network"""
    def __init__(self, num_classes=1000):
        super(InvDynNet, self).__init__()
        # shared resnet34 weights for both heads of the net ...
        self.conv = resnet34(pretrained=True, num_classes=num_classes)
        self.fc1 = nn.Linear(2000, 2) # pick point
        self.fc2 = nn.Linear(2002, 2) # place point (conditional on pick)

    def forward(self, x):
        """
        Assume x is given as N x 2 x C x H x W for before/after images
        """
        x1 = x[:,0,:,:,:]
        x2 = x[:,1,:,:,:]
        x1 = self.conv(x1) # N x 1000
        x2 = self.conv(x2) # N x 1000
        x = torch.cat((x1, x2), 1) # N x 2000
        x = F.relu(x)
        pick = self.fc1(x)
        pick = F.sigmoid(pick)
        place = self.fc2(torch.cat((x, pick), 1))
        place = F.sigmoid(place)
        return torch.cat((pick, place), 1) # N x 4

class ForDynNet(nn.Module):
    """CRL network"""
    def __init__(self, num_classes=1000):
        super(ForDynNet, self).__init__()
        self.conv = resnet34(pretrained=True, num_classes=num_classes)
        self.fc1 = nn.Linear(2000, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = self.conv(s) # N x 1000
        x = torch.cat((s, a), 1) # N x 2000
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.tanh(x) # output can be negative.


