

import torch
import random
import itertools
import numpy as np
from scipy.spatial import distance
from thundersvm import SVC, OneClassSVM
#from sklearn.svm import OneClassSVM, SVC

from agent.dynamics.mpc import MPC

from agent.ppo.PPOAgentClass import PPOAgent

from collections import deque


class GlobalOption(Option):
    def __init__(self, *, name, state_dim, action_dim, buffer_length,
                 timeout, max_steps, device, option_idx, lr_c, lr_a, path_to_model=None):
        self.name = name
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.device = device
        # self.use_vf = use_vf
        # self.global_solver = global_solver
        # self.use_global_vf = use_global_vf
        self.timeout = timeout
        self.steps = 0
        self.max_steps = max_steps
        self.global_init = True
        self.buffer_length = buffer_length

        self.global_option = None
        self.is_global_option = True

        self.seed = 0
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period


        self.state_dim = state_dim
        self.action_dim = action_dim
        self.solver = PPOAgent(obs_n_channels=self.state_dim + 1, n_actions=self.action_dim, device_id=-1)

        self.children = []
        self.success_curve = []
        self.effect_set = []

        if path_to_model:
            print(f"Loading model from {path_to_model} for {self.name}")
            self.solver.load_model(path_to_model)

        print(f"Created global option {self.name} with option_idx={self.option_idx}")


    def get_training_phase(self):
        return "initiation_done"

    def is_in_training_phase(self):
        return False

    def extract_state_features(self, state):
        return state

    def is_init_true(self, state):
        return True

    def is_in_term_set(self, state):
        return True

    def is_term_true(self, state):
        return True

    def pessimistic_is_init_true(self, state):
        return True


    def act(self, state):
        """ Epsilon-greedy action selection. """
        self.steps += 1
        self.num_executions += 1

        if random.random() < self._get_epsilon():
            return np.random.randint(0, self.action_dim)

        return self.solver.act(state)

    def update_model(self, state, action, reward, next_state, done):
        """ Learning update for option model/actor/critic. """
        self.solver.step(self.extract_state_features(state), action, reward, self.extract_state_features(next_state), done)
