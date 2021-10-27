from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import norm
import matplotlib.pyplot as plt

from model import RNDModel



class RNDAgent:
    def __init__(self, rnd_model, use_cuda=False):
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.rnd = rnd_model
        self.rnd = self.rnd.to(self.device)

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float)

        predict_next_feature, target_next_feature = self.rnd(obs)
        output = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return output.data.cpu().numpy()

class DeviationSubgoalGenerator:
    def __init__(self, nov_rnd, use_cuda=False):
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.nov_rnd = RNDAgent(nov_rnd, use_cuda=use_cuda)

        self.state_buff = deque(maxlen=25000)

    def train_gaussian(self, traj):
        self.state_buff.extend(traj)

    def get_nov_vals_for_gaussian(self, traj, obs_rms):
        norm_traj = ((traj - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
        traj_novelties = self.nov_rnd.forward(norm_traj)
        return traj_novelties

    def get_gaussian_nov_state(self, traj, obs_rms):
        norm_traj = ((traj - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
        norm_buff = ((self.state_buff - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)

        traj_novelties = self.nov_rnd.forward(norm_traj)
        buff_novelties = self.nov_rnd.forward(norm_buff)

        mean, std = norm.fit(buff_novelties)

        STD = abs(traj_novelties - mean) / std
        I = np.where(STD > 2)[0]
        if len(I) == 0:
            return [], [], [], []

        return I, traj_novelties[I], traj[I], STD[I]

    def generate_subgoal_index(self, traj, obs_rms):
        norm_traj = ((traj - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
        norm_buff = ((self.state_buff - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)

        traj_novelties = self.nov_rnd.forward(norm_traj)
        buff_novelties = self.nov_rnd.forward(norm_buff)

        mean, std = norm.fit(buff_novelties)

        max_nov_index = np.argmax(traj_novelties)
        max_nov = traj_novelties[max_nov_index]
        print(max_nov)
        if (max_nov - mean) / std > 3:
            return max_nov_index
        return None #no subgoal for this trajectory
