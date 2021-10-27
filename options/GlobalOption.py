from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agents import *
from config import *
from envs import *
from utils import *

class GlobalOption:

    def __init__(self, default_config, run_path, AgentType=RNDAgent):
        self.train_method = default_config['TrainMethod']
        assert train_method == 'RND'
        self.env_id = default_config['EnvID']
        self.env_type = default_config['EnvType']

        if self.env_type == 'atari':
            env = gym.make(self.env_id)
        else:
            raise NotImplementedError
        self.input_size = env.observation_space.shape  # 4
        self.output_size = env.action_space.n  # 2

        if 'Breakout' in self.env_id:
            output_size -= 1

        env.close()

        self.log_path = run_path / 'logs'
        self.log_path.mkdir()

        self.use_cuda = default_config.getboolean('UseGPU')
        self.use_gae = default_config.getboolean('UseGAE')
        self.use_noisy_net = default_config.getboolean('UseNoisyNet')

        self.lam = float(default_config['Lambda'])
        self.num_envs = int(default_config['NumEnv'])

        self.num_step = int(default_config['NumStep'])

        self.ppo_eps = float(default_config['PPOEps'])
        self.epoch = int(default_config['Epoch'])
        self.mini_batch = int(default_config['MiniBatch'])
        self.batch_size = int(self.num_step * self.num_envs / self.mini_batch)
        self.learning_rate = float(default_config['LearningRate'])
        self.entropy_coef = float(default_config['Entropy'])
        self.gamma = float(default_config['Gamma'])
        self.int_gamma = float(default_config['IntGamma'])
        self.clip_grad_norm = float(default_config['ClipGradNorm'])
        self.ext_coef = float(default_config['ExtCoef'])
        self.int_coef = float(default_config['IntCoef'])

        self.sticky_action = default_config.getboolean('StickyAction')
        self.action_prob = float(default_config['ActionProb'])
        self.life_done = default_config.getboolean('LifeDone')

        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))

        self.discounted_reward = RewardForwardFilter(self.int_gamma)


        # for initialization phase
        self.next_obs = []

        # for n-step rollout phase
        self.states = np.zeros([self.num_envs, 4, 84, 84])

        self.current_step = 0
        self.num_steps_in_rollout = 128

        self.total_next_obs = []
        self.total_int_reward = []
        self.total_state = []
        self.total_reward = []
        self.total_done = []
        self.total_action = []
        self.total_ext_values = []
        self.total_int_values = []
        self.total_policy = []
        self.total_policy_np = []


        self.agent = AgentType(
            self.input_size,
            self.output_size,
            self.num_envs,
            self.num_step,
            self.gamma,
            lam=self.lam,
            learning_rate=self.learning_rate,
            ent_coef=self.entropy_coef,
            clip_grad_norm=self.clip_grad_norm,
            epoch=self.epoch,
            batch_size=self.batch_size,
            ppo_eps=self.ppo_eps,
            use_cuda=self.use_cuda,
            use_gae=self.use_gae,
            use_noisy_net=self.use_noisy_net
        )

    def get_intrinsic_reward(self, state):
        '''
        Perform obs_rms initialization

        Args:
            state (np.array): A single observation (4, 84, 84)

        Returns:
            None
        '''
        return self.agent.compute_intrinsic_reward(
                    ((state[-1, :, :].reshape([1, 84, 84]) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5))

    def init_update(self, state):
        '''
        Perform obs_rms initialization

        Args:
            state (np.array): A single observation (4, 84, 84)

        Returns:
            None
        '''
        self.next_obs.append(state[-1, :, :].reshape([1, 84, 84]))
        if len(self.next_obs) % (self.num_step * self.num_envs) == 0:
            self.next_obs = np.stack(self.next_obs)
            self.obs_rms.update(self.next_obs)
            self.next_obs = []


    def act(self):
        '''
        Acts based on agent policy

        Args:
            None

        Returns:
            None
        '''
        actions, value_ext, value_int, policy = self.agent.get_action(np.float32(self.states) / 255.)
        return actions, value_ext, value_int, policy



    def update(self, actions, next_states, next_obs, rewards, dones, real_dones, policy, value_ext, value_int):
        '''
        Perform update for a single step of n-step rollout

        Args:
            actions (np.array): Vector of actions (num_envs, )
            next_states (np.array): Vector of next_states (num_envs, 4, 84, 84)
            next_obs (np.array): Vector of next_obs (num_envs, 1, 84, 84)
            rewards (np.array): Vector of rewards (num_envs, )
            dones (np.array): Vector of dones (num_envs, )
            real_dones (np.array): Vector of real_dones (num_envs, )

        Returns:
            None
        '''
        next_states = np.stack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)
        real_dones = np.hstack(real_dones)
        next_obs = np.stack(next_obs)

        # total reward = int reward + ext Reward
        intrinsic_reward = self.agent.compute_intrinsic_reward(
            ((next_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5))
        intrinsic_reward = np.hstack(intrinsic_reward)

        self.total_next_obs.append(next_obs)
        self.total_int_reward.append(intrinsic_reward)
        self.total_state.append(self.states)
        self.total_reward.append(rewards)
        self.total_done.append(dones)
        self.total_action.append(actions)
        self.total_ext_values.append(value_ext)
        self.total_int_values.append(value_int)
        self.total_policy.append(policy)
        self.total_policy_np.append(policy.cpu().numpy())

        self.states = next_states[:, :, :, :]

        self.current_step += 1

        if self.current_step >= self.num_steps_in_rollout:
            self.perform_PPO_RND_update()
            self.current_step = 0

    def perform_PPO_RND_update(self):
        '''
        Perform update at the end of the n-step rollout

        Args:
            None

        Returns:
            None
        '''
        _, value_ext, value_int, _ = self.agent.get_action(np.float32(self.states) / 255.)
        self.total_ext_values.append(value_ext)
        self.total_int_values.append(value_int)

        self.total_state = np.stack(self.total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        self.total_reward = np.stack(self.total_reward).transpose().clip(-1, 1)
        self.total_action = np.stack(self.total_action).transpose().reshape([-1])
        self.total_done = np.stack(self.total_done).transpose()
        self.total_next_obs = np.stack(self.total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        self.total_ext_values = np.stack(self.total_ext_values).transpose()
        self.total_int_values = np.stack(self.total_int_values).transpose()
        self.total_logging_policy = np.vstack(self.total_policy_np)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        self.total_int_reward = np.stack(self.total_int_reward).transpose()
        #total_int_reward = np.stack(total_int_reward).swapaxes(0, 1)
        total_reward_per_env = np.array([self.discounted_reward.update(reward_per_step) for reward_per_step in self.total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        self.reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        self.total_int_reward /= np.sqrt(self.reward_rms.var)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(self.total_reward,
                                              self.total_done,
                                              self.total_ext_values,
                                              self.gamma,
                                              self.num_step,
                                              self.num_envs)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(self.total_int_reward,
                                              np.zeros_like(self.total_int_reward),
                                              self.total_int_values,
                                              self.int_gamma,
                                              self.num_step,
                                              self.num_envs)

        # add ext adv and int adv
        total_adv = int_adv * self.int_coef + ext_adv * self.ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        self.obs_rms.update(self.total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        self.agent.train_model(np.float32(self.total_state) / 255., ext_target, int_target, self.total_action,
                          total_adv, ((self.total_next_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5),
                          self.total_policy)

        self.total_next_obs = []
        self.total_int_reward = []
        self.total_state = []
        self.total_reward = []
        self.total_done = []
        self.total_action = []
        self.total_ext_values = []
        self.total_int_values = []
        self.total_policy = []
        self.total_policy_np = []
