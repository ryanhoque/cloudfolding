import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from .core import InvDynNet, ForDynNet
from .dataset import InvDynamicsDataset, ForwardDynamicsDataset
import os
from torch.utils.data import DataLoader

def forward(batch, model, forward_dynamics=False):
    lossfn = nn.MSELoss()
    if forward_dynamics:
        imgs, kps, labels = batch
        loss = lossfn(model(imgs, kps).squeeze(), labels) 
    else:
        imgs, kps = batch
        loss = lossfn(model(imgs), kps)
    return loss

def train(device_idx=1, seed=42, train_data=DATADIR+'/train2', 
    test_data=DATADIR+'/test2',
    output_dir='pyreach/tools/inv_dynamics/checkpoints', epochs=100, batch_size=16, forward_dynamics=False):
    # setup
    device = torch.device("cuda", device_idx)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device_idx)
    if forward_dynamics:
        model = ForDynNet().to(device)
    else:
        model = InvDynNet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # load data 
    if forward_dynamics:
        train_dataset = ForwardDynamicsDataset(train_data, aug=False)
        test_dataset = ForwardDynamicsDataset(test_data, aug=False)
    else:
        train_dataset = InvDynamicsDataset(train_data, aug=False)
        test_dataset = InvDynamicsDataset(test_data, aug=False)
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # fit
    best_loss = 9999999
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for epoch in range(epochs):
        train_loss = 0.0
        for i_batch, batch in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(batch, model, forward_dynamics)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if epoch == 0:
                print('batch #{} loss {}'.format(i_batch, loss.item()))
        print('epoch {} train loss {}'.format(epoch, train_loss/i_batch))
        test_loss = 0.0
        for i_batch, batch in enumerate(test_data):
            with torch.no_grad():
                loss = forward(batch, model, forward_dynamics)
                test_loss += loss.item()
        print('epoch {} test loss {}'.format(epoch, test_loss/i_batch))
        if test_loss < best_loss:
            print('new best loss!')
            best_loss = test_loss
            prefix = 'forward' if forward_dynamics else 'inverse'
            torch.save(model.state_dict(), os.path.join(output_dir, '{}-tmp.pth'.format(prefix)))

if __name__ == '__main__': # train
    train(forward_dynamics=True)
