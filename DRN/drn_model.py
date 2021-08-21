from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import RNDModel

import time

import queue


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float("inf"):
        total_norm = max(p.grad.data.abs().max().item() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


class RNDAgent:
    def __init__(
        self,
        rnd_model,
        learning_rate=1e-4,
        epoch=3,
        batch_size=128,
        update_proportion=0.25,
        use_cuda=False,
    ):
        self.epoch = epoch
        self.batch_size = batch_size
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.rnd = rnd_model
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=learning_rate)

        self.rnd = self.rnd.to(self.device)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.rnd.predictor.apply(init_weights)
        self.rnd.target.apply(init_weights)

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float)

        predict_next_feature, target_next_feature = self.rnd(obs)
        output = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return output.data.cpu().numpy()

    def train(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)

        sample_range = np.arange(len(obs))
        forward_mse = nn.MSELoss(reduction='none')

        for _ in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(len(obs) // self.batch_size):
                sample_idx = sample_range[self.batch_size * j : self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(obs[sample_idx])

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)) < self.update_proportion
                loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1]))
                # ---------------------------------------------------------------------------------

                self.optimizer.zero_grad()
                loss.backward()
                global_grad_norm_(self.rnd.predictor.parameters())
                self.optimizer.step()


class DeepRelNov:
    def __init__(
        self,
        nov_rnd,
        train_nov_rnd,
        input_size,
        output_size,
        use_cuda=False,
        rel_nov_percentile=95,
        freq_percentile=50,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.n_l = 7
        self.thresh_lr = 0.1

        self.rel_nov_percentile = rel_nov_percentile
        self.rel_nov_thresh = 100#100

        self.train_nov_rnd = train_nov_rnd
        if self.train_nov_rnd:
            self.state_buf = deque(maxlen=500)
        if nov_rnd is None:
            nov_rnd = RNDModel(input_size, output_size)
        self.nov_rnd = RNDAgent(nov_rnd, use_cuda=use_cuda)

        self.freq_percentile = freq_percentile
        self.freq_thresh = 0
        self.freq_rnd = RNDAgent(RNDModel(input_size, output_size), use_cuda=use_cuda)

        self.rel_nov_state_buf = deque(maxlen=100)

        self.last_trajectory = None
        # self.last_full_state_trajectory = None

        self.training_buffer = []

        self.training_iterations = 0
        self.pretraining_duration = 50


    def is_drn_trained(self):
        return self.training_iterations >= self.pretraining_duration

    # def add_to_train_buffer(self, trajectory):
    #     self.training_buffer.append(trajectory)
    #     self.last_trajectory = trajectory
    #
    # def train_from_buffer(self):
    #     for traj in self.training_buffer:
    #         self.train_rel_nov(traj)
    #     self.training_buffer = []

    def train_rel_nov(self, trajectory):
        self.last_trajectory = trajectory
        # self.last_full_state_trajectory = full_state_trajectory
        self.training_iterations += 1
        if self.train_nov_rnd:
            self.state_buf.extend(trajectory)
            self.nov_rnd.train(self.state_buf)
        nov_vals = self.get_nov_vals(trajectory)
        rel_nov_vals = self.get_rel_nov_vals(trajectory, nov_vals)
        self.update_rel_nov_thresh(rel_nov_vals)

        rel_nov_states, _ = self.get_rel_nov_states(rel_nov_vals, trajectory)
        self.rel_nov_state_buf.extend(rel_nov_states)

        # TODO: should we train even if there are no rel_nov_states?
        self.freq_rnd.train(self.rel_nov_state_buf)

        if len(rel_nov_states) > 0:
            freq_vals = self.freq_rnd.forward(rel_nov_states)
            self.update_freq_thresh(freq_vals)


        if self.training_iterations > 1000 and self.training_iterations % 50 == 0:
            self.get_rel_nov_subgoals(trajectory, plot=True)


    def train_max(self, trajectory):
        nov_vals = self.get_nov_vals(trajectory)
        I = self.max_so_far(nov_vals)
        nov_states = trajectory[I]

        self.rel_nov_state_buf.extend(nov_states)

        # TODO: should we train even if there are no rel_nov_states?
        self.freq_rnd.train(self.rel_nov_state_buf)

        if len(nov_states) > 0:
            freq_vals = self.freq_rnd.forward(nov_states)
            self.update_freq_thresh(freq_vals)

    def update_rel_nov_thresh(self, rel_nov_vals):
        p = np.percentile(rel_nov_vals, self.rel_nov_percentile)
        self.rel_nov_thresh = self.thresh_lr * p + (1 - self.thresh_lr) * self.rel_nov_thresh

    def update_freq_thresh(self, freq_vals):
        p = np.percentile(freq_vals, self.freq_percentile)
        self.freq_thresh = self.thresh_lr * p + (1 - self.thresh_lr) * self.freq_thresh

    def get_nov_vals(self, trajectory):
        return self.nov_rnd.forward(trajectory)

    def get_rel_nov_vals(self, trajectory, nov_vals):
        "Gets the relative novelty values for the states in the trajectory"

        def get_rel_nov(i):
            if i < self.n_l or i >= len(trajectory) - self.n_l:
                return 0

            visits_before = nov_vals[i - self.n_l : i]
            visits_after = nov_vals[i : i + self.n_l]

            # return np.sum(visits_after) / np.sum(visits_before)
            return max(np.sum(visits_after) - np.sum(visits_before),0)

        return np.array([get_rel_nov(i) for i in range(len(trajectory))])

    def get_rel_nov_states(self, rel_nov_vals, trajectory):
        "Gets states in the trajectory whose relative novelty is greater than the novelty threshold"
        I = np.arange(len(trajectory))[rel_nov_vals > self.rel_nov_thresh]
        return trajectory[I], I

    def max_so_far(self, nov_vals):
        M = np.maximum.accumulate(nov_vals)
        V = M[1:] > M[:-1]
        I = [x + 1 for x in np.where(V[1:] < V[:-1])[0]]
        return I[1:]

    def get_latest_subgoals(self):
        return self.get_subgoals(self.last_trajectory)

    def get_subgoals(self, trajectory):
        # return self.get_max_subgoals(trajectory)
        return self.get_rel_nov_subgoals(trajectory)

    def get_filtered_subgoals(self, trajectory, N):
        subgoals, _, _, freq_vals, _ = self.get_subgoals(trajectory)
        N = 1
        top_N = np.argpartition(freq_vals, -N)[-N:]
        return subgoals[top_N]

    def get_max_subgoals(self, trajectory):
        nov_vals = self.get_nov_vals(trajectory)
        I = np.array(self.max_so_far(nov_vals))
        nov_states = trajectory[I]

        freq_vals = self.freq_rnd.forward(nov_states)
        I_freq = np.arange(len(freq_vals))[freq_vals < self.freq_thresh]
        if len(I) == 0 or len(I_freq) == 0:
            return [], [], [], [], []

        I = I[np.array(I_freq)]
        return nov_states[I_freq], nov_vals[I], None, freq_vals[I_freq], I

    def get_rel_nov_subgoals(self, trajectory, plot=False):
        nov_vals = self.get_nov_vals(trajectory)
        rel_nov_vals = self.get_rel_nov_vals(trajectory, nov_vals)

        #I_nov is indexes of rel_nov states in original trajectory
        rel_nov_states, I = self.get_rel_nov_states(rel_nov_vals, trajectory)
        if len(rel_nov_states) == 0:
            return [], [], [], [], []

        freq_vals = self.freq_rnd.forward(rel_nov_states)
        #I is relative to list of relatively novel states not global traj
        I_freq = np.arange(len(freq_vals))[freq_vals < self.freq_thresh]
        if len(I) == 0:
            return [], [], [], [], []


        I = I[I_freq]

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(nov_vals)
            plt.plot(rel_nov_vals)
            plt.title(I[np.argmax(freq_vals[I_freq])])
            plt.savefig("nov-vals" + str(I[np.argmax(freq_vals[I_freq])]))

            states = rel_nov_states[I_freq]
            fig, axs = plt.subplots(len(state))
            for i,state in enumerate(states):
                axs[i].imshow(state)
                axs[i].set_title(I_freq[i])
            plt.savefig("nov-states" + str(I[np.argmax(freq_vals[I_freq])]))

            # for state in trajectory:
            #     plt.imshow(np.sum(state, axis=0))
            #     plt.draw()
            #     plt.pause(.01)
            #     plt.clf()

        return rel_nov_states[I_freq], nov_vals[I], rel_nov_vals[I], freq_vals[I_freq], I
