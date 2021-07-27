import random
import numpy as np

import torch
from torch import nn

import pfrl
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead

class PPOAgent(object):
    def __init__(self,
                 obs_n_channels,
                 n_actions,
                 lr=2.5e-4,
                 update_interval=128*8,
                 batchsize=32*8,
                 epochs=4,
                 device_id=0):

        def lecun_init(layer, gain=1):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                pfrl.initializers.init_lecun_normal(layer.weight, gain)
                nn.init.zeros_(layer.bias)
            else:
                pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
                pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
                nn.init.zeros_(layer.bias_ih_l0)
                nn.init.zeros_(layer.bias_hh_l0)
            return layer

        self.model = nn.Sequential(
            lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
            nn.ReLU(),
            lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            lecun_init(nn.Linear(3136, 512)),
            nn.ReLU(),
            pfrl.nn.Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(512, n_actions), 1e-2),
                    SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(512, 1)),
            ),
        )

        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if device_id > -1 else "cpu")
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        def phi(x):
            # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255.

        self.agent = PPO(self.model,
                         opt,
                         gpu=device_id,
                         phi=phi,
                         update_interval=update_interval,
                         minibatch_size=batchsize,
                         epochs=epochs,
                         clip_eps=0.1,
                         clip_eps_vf=None,
                         standardize_advantages=True,
                         entropy_coef=1e-2,
                         recurrent=False,
                         max_grad_norm=0.5,
        )
        self.agent._initialize_batch_variables(1)

    def act(self, obs):
        return self.agent.act(obs)

    def step(self, obs, action, reward, next_obs, done):
        assert obs is not None and next_obs is not None
        # if done:
        #     import pdb; pdb.set_trace()
        self.agent.batch_last_state = [obs]
        self.agent.batch_last_action = [action]
        self.agent.observe(next_obs, reward, done, reset=False)

    def get_batched_qvalues(self, states):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        device = f"cuda:{self.device_id}" if self.device_id >= 0 else "cpu"
        states = states.to(device)
        with torch.no_grad():
            action_values = self.agent.model(states)
        return action_values.q_values
