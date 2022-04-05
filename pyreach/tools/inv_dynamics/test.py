import torch
from .core import InvDynNet, ForDynNet
from torch.utils.data import DataLoader
from pyreach.tools.analytic_benchmark import random_cloth_pick, random_workspace_point
import matplotlib.pyplot as plt
import sys
import cv2
import os
import numpy as np
import time
import scipy.stats as stats

class ForwardInference(): # CRL
    def __init__(self, model_path, device_idx=1):
        device = torch.device("cuda", device_idx)
        self.model = ForDynNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        torch.cuda.set_device(device_idx)
        # CEM params
        self.popsize = 2000
        self.num_elites = 400
        self.alpha = 0.1
        self.ac_lb = np.array([0.,0.,0.,0.])
        self.ac_ub = np.array([1.,1.,1.,1.])
        self.act_dim = 4
        self.batch_size = 100
        self.mean = np.array([0.5,0.5,0.5,0.5])
        self.var = np.square(self.ac_ub - self.ac_lb) / 16
        self.num_iters = 10

    def random_sampling(self, obs):
        # preprocess obs
        raw_img = obs
        obs = torch.from_numpy(obs.copy()).float() / 255.
        obs = obs[:,150:510,:]
        batch_size = 100
        obs = obs.permute(2,0,1).repeat(batch_size,1,1,1).cuda()
        max_reward = -999
        timestamp = time.time()
        for i in range(0,5000,batch_size):
            with torch.no_grad():
                act = torch.zeros((batch_size, 1000))
                for j in range(batch_size):
                    pick_x, pick_y = random_cloth_pick(raw_img)
                    place_x, place_y = random_workspace_point(pick=(pick_x, pick_y))
                    pick_x = max(min((pick_x - 150) / 360., 1.), 0.)
                    pick_y = max(min((pick_y - 0) / 360., 1.), 0.)
                    place_x = max(min((place_x - 150) / 360., 1.), 0.)
                    place_y = max(min((place_y - 0) / 360., 1.), 0.)
                    act[j] = torch.tensor([pick_x, pick_y, place_x, place_y]).tile((250,))
                rewards = self.model(obs, act.float().cuda()).cpu().numpy()[:,0]
            if max(rewards) > max_reward:
                max_act = act[np.argmax(rewards),:4].numpy()
                max_reward = max(rewards)
        print('Found act with reward {} in time {}'.format(max_reward, time.time() - timestamp))
        act = (max_act * 360.) + np.array([150,0,150,0])
        act = act.astype(np.uint16)
        return {'pick': (act[0], act[1]), 'place': (act[2], act[3])}

    def cem(self, obs):
        """run 1-step CEM"""
        timestamp = time.time()
        mean, var = self.mean.copy(), self.var.copy()
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))
        # preprocess obs
        obs = torch.from_numpy(obs.copy()).float() / 255.
        obs = obs[:,150:510,:]
        obs = obs.permute(2,0,1).repeat(100,1,1,1).cuda()
        for i in range(self.num_iters):
            lb_dist, ub_dist = mean - self.ac_lb, self.ac_ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            samples = X.rvs(size=[self.popsize, self.act_dim]) * np.sqrt(constrained_var) + mean
            with torch.no_grad():
                costs = []
                for batch in range(0, self.popsize // self.batch_size):
                    acts = torch.from_numpy(samples[batch*self.batch_size:(batch+1)*self.batch_size]).float().cuda()
                    acts = acts.tile((1,250))
                    costs.extend(-self.model(obs, acts).cpu().numpy().squeeze())
            print('CEM Iteration {} Cost {} +/- {}'.format(i, np.mean(costs), np.std(costs)))
            elites = samples[np.argsort(costs)][:self.num_elites]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            mean, var = self.alpha * mean + (1 - self.alpha) * new_mean, self.alpha * var + (1 - self.alpha) * new_var
        self.mean = mean.copy() # set next mean to be the current action
        # postprocess act
        print('time {}'.format(time.time() - timestamp))
        return (mean * 360.) + np.array([150,0,150,0])

    def predict(self, obs, act):
        # input: (single) obs and act
        # output: predicted delta coverage (normalized)
        # process img
        assert len(obs.shape) == 3
        h, w = obs.shape[0], obs.shape[1]
        obs = torch.from_numpy(obs.copy()).float() / 255.
        obs = obs[:, 150:510, :]
        obs = obs.permute(2, 0, 1)
        obs = obs.unsqueeze(dim=0).cuda()
        # process act
        act_ = np.zeros(4)
        act_[0] = max(min((act[0] - 150) / 360., 1.), 0.)
        act_[1] = max(min((act[1] - 0) / 360., 1.), 0.)
        act_[2] = max(min((act[2] - 150) / 360., 1.), 0.)
        act_[3] = max(min((act[3] - 0) / 360., 1.), 0.)
        act = torch.tensor(act_).unsqueeze(dim=0).float().tile((250,)).cuda()
        # infer
        with torch.no_grad():
            pred = self.model(obs, act).cpu().numpy()[0][0]
        return pred * 50000.

    def viz(self, obs, act):
        img = obs.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_x1, gt_y1 = int(act['pick'][0]), int(act['pick'][1])
        gt_x2, gt_y2 = int(act['place'][0]), int(act['place'][1])
        print('act', act)
        cv2.circle(img, (gt_x1, gt_y1), 5, (0,128,0), -1)
        cv2.circle(img, (gt_x2, gt_y2), 5, (0,255,0), -1)
        cv2.imwrite('test.png', img)

class InvInference(): # IDYN
    def __init__(self, model_path, device_idx=1):
        device = torch.device("cuda", device_idx)
        self.model = InvDynNet().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        torch.cuda.set_device(device_idx)

    def get_closest_goal(self, obs, goal):
        # TODO test/debug this function
        # perform translations and rotations to get closest 'version' of goal img
        # mask the goal image to pass in for matching, obs will get masked inside
        location_mask = np.zeros(obs.shape[:2], dtype=obs.shape)
        location_mask[WORKSPACE_Y_BOUNDS[0]:WORKSPACE_Y_BOUNDS[1], WORKSPACE_X_BOUNDS[0]:WORKSPACE_X_BOUNDS[1]] = 1
        goal_masked = location_mask*(goal[:, :, 2] > goal[:, :, 0])

        plt.imshow(goal_masked)
        plt.show(goal_masked)

        # find the best fitting rotation using the match image function
        img_center, mask_center, best_angle, best_mask, score = match_image(obs*location_mask, goal_masked)

        # now manually transform the original goal image ourselves, since the best_mask returned would be the masked goal image
        shifted_goal = shift(target_mask, (int(img_center[0] - mask_center[0]), int(img_center[1] - mask_center[1])))
        shifted_rotated_goal = rotate(shifted_goal, best_angle, center=(img_center[1], img_center[0]))

        plt.imshow(shifted_goal)
        plt.show()

        return shifted_rotated_goal

    def viz(self, obs, act, obs2, index=0):
        img = obs.copy()
        img2 = obs2.copy()
        pred = self.predict(obs, obs2, closest=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        gt_x1, gt_y1 = int(act[0]), int(act[1])
        gt_x2, gt_y2 = int(act[2]), int(act[3])
        p_x1, p_y1 = pred['pick']
        p_x2, p_y2 = pred['place']
        print(p_x1, p_y1, p_x2, p_y2)
        print('pred', pred)
        print('gt', act)
        cv2.circle(img, (gt_x1, gt_y1), 5, (0,128,0), -1)
        cv2.circle(img, (gt_x2, gt_y2), 5, (0,255,0), -1)
        cv2.circle(img, (p_x1, p_y1), 5, (0,0,128), -1)
        cv2.circle(img, (p_x2, p_y2), 5, (0,0,255), -1)
        imgs = np.hstack((img, img2))
        cv2.imwrite('{}.png'.format(index), imgs)

    def predict(self, obs, goal, closest=False):
        # input: current obs and goal image
        # output: action to execute in pixel coords

        # process obs & goal
        if closest:
            goal = self.get_closest_goal(obs, goal)
        h, w = obs.shape[0], obs.shape[1]
        obs = torch.from_numpy(obs.copy()).float() / 255.
        obs = obs[:, 150:510, :]
        obs = obs.permute(2, 0, 1)
        goal = torch.from_numpy(goal.copy()).float() / 255.
        goal = goal[:, 150:510, :]
        goal = goal.permute(2, 0, 1)
        if len(obs.shape) == 3:
            input_data = torch.stack([obs, goal]).unsqueeze(dim=0).cuda()
        # infer
        with torch.no_grad():
            pred = self.model(input_data).cpu().numpy()
        # process action
        pred = pred[0] * 360.
        pred[0] = pred[0] + 150
        pred[1] = pred[1] + 0
        pred[2] = pred[2] + 150
        pred[3] = pred[3] + 0
        pred = pred.astype(np.uint16)
        return {'pick': (pred[0], pred[1]), 'place': (pred[2], pred[3])}

if __name__ == '__main__':
    DATADIR = 'data'
    inf = InvInference(DATADIR+'/flatten-idyn.pth', 1)
    test_data = DATADIR+'/test'
    goal = np.load(DATADIR+'/smooth.npy')
    files = []
    for filename in sorted(os.listdir(test_data)):
        files.append(os.path.join(test_data, filename))
    for index in range(len(files)):
        img_file = files[index]
        img, img2, act = np.load(img_file, allow_pickle=True)
        img2 = goal.copy()
        inf.viz(img, act, img2, index)
    sys.exit(0)
    inf = ForwardInference(DATADIR+'/flatten-crl.pth')
    test_data=DATADIR+'/test2'
    imgs = []
    for filename in sorted(os.listdir(test_data)):
        imgs.append(os.path.join(test_data, filename))
    img_file = imgs[0]
    img, act, dc = np.load(img_file, allow_pickle=True)
    act = inf.random_sampling(img)
    #act = inf.cem(img)
    inf.viz(img, act)
    vals = []
    for index in range(len(imgs)):
        img_file = imgs[index]
        img, act, dc = np.load(img_file, allow_pickle=True)
        pred = inf.predict(img, act)
        vals.append(pred - dc*50000)
    vals = [abs(v) for v in vals]
    print('average diff {} pixels or {}%'.format(sum(vals)/len(vals), sum(vals)/len(vals)/500.))
    print('std diff {} pixels or {}%'.format(np.array(vals).std(), np.array(vals).std()/500.))
    print('min diff {} pixels or {}%'.format(min(vals), min(vals)/500.))
    print('max diff {} pixels or {}%'.format(max(vals), max(vals)/500.))
