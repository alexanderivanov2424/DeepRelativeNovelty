from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe
from collections import deque

from agents import *
from config import *
from DRN.OptionHandler import OptionHandler
from DRN.drn_model import DeepRelNov
from DRN.DRNAgent import DRNAgent
from envs import *
from utils import *

from guppy import hpy
import gc

import sys
import resource

def memory_limit(n = 2):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / n, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def dump_garbage():
    gc.collect()

    print("\nGARBAGE OBJECTS:")
    for x in gc.garbage:
        s = str(x)
        if len(s) > 80:
            s = s[:80]
        print(type(x),"\n  ", s)

def main():

    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    assert train_method == 'RND'
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    run_path = Path(f'runs/{env_id}_{datetime.now().strftime("%b%d_%H-%M-%S")}')
    log_path = run_path / 'logs'

    run_path.mkdir(parents=True)
    log_path.mkdir()

    writer = SummaryWriter(log_path)

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if use_cuda else 'torch.FloatTensor')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 4, 84, 84)) #RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    else:
        raise NotImplementedError

    # agent = agent(
    #     input_size,
    #     output_size,
    #     num_worker,
    #     num_step,
    #     gamma,
    #     lam=lam,
    #     learning_rate=learning_rate,
    #     ent_coef=entropy_coef,
    #     clip_grad_norm=clip_grad_norm,
    #     epoch=epoch,
    #     batch_size=batch_size,
    #     ppo_eps=ppo_eps,
    #     use_cuda=use_cuda,
    #     use_gae=use_gae,
    #     use_noisy_net=use_noisy_net
    # )
    input_size = (4, 84, 84)

    agent = DRNAgent(input_size, output_size)
    drn_model = DeepRelNov(agent.rnd, True, input_size, output_size, use_cuda=use_cuda)
    option_handler = OptionHandler(drn_model, len(input_size), output_size, torch.device)



    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            agent.rnd.target.load_state_dict(torch.load(target_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
        print('load finished!')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    current_option = [None for i in range(num_worker)]
    option_trajectories = [[] for i in range(num_worker)]
    option_duration = [0 for i in range(num_worker)]

    trajectories = [[] for i in range(num_worker)]

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    for _ in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for i,parent_conn in enumerate(parent_conns):
            s, r, d, rd, lr, _ = parent_conn.recv()
            # next_obs.append(s[-1, :, :].reshape([1, 84, 84]))
            if rd:
                traj = ((np.stack(trajectories[i]) - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
                drn_model.train_rel_nov(traj)
                trajectories[i].clear()
            next_obs.append(s)
            trajectories[i].append(s)

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End to initalize...')


    # episode_rewards = [[] for _ in range(num_worker)]
    # step_rewards = [[] for _ in range(num_worker)]
    global_ep = 0


    while True:

        # print("[+]", get_size(agent))
        # print("[+]", get_size(drn_model)/1000000, "MB")

        gc.collect()

        total_state, total_reward, total_done, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], []


        global_step += (num_worker * num_step)
        global_update += 1

        # if len(option_handler.options) > 4:
        #     print(option_handler.options)
        #     exit()
        # if global_update > 100000:
        #     print(option_handler.options)
        #     exit()


        # Step 1. n-step rollout
        for cur_step in range(num_step):
            actions = [None for i in range(len(parent_conns))]
            for i, parent_conn in enumerate(parent_conns):

                if current_option[i] is None:
                    current_option[i] = agent.act(states[i], option_handler)
                    assert np.all(np.isfinite(states[i]))
                    action = current_option[i].act(states[i])
                else:
                    action = current_option[i].act(states[i])
                actions[i] = action
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for i, parent_conn in enumerate(parent_conns):
                s, r, d, rd, lr, info = parent_conn.recv()
                # obs = np.float32(s[-1, :, :].reshape([1, 1, 84, 84])) / 255.

                rd = d or rd

                s_norm = ((np.array([trajectories[i][-1]]) - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
                option_terminated = current_option[i].is_term_true(s_norm)

                # print("updating option", current_option[i])
                if current_option[i].is_global_option:
                    int_reward = agent.compute_intrinsic_reward([s])
                else:
                    int_reward = 0
                #TODO do we need to reward normal options for hitting termination set
                option_handler.update(states[i], actions[i], r + int_reward[0], s, option_terminated, current_option[i])
                do_option_update = False

                if option_terminated:
                    if not current_option[i].is_global_option:
                        print("hit goal by option:", current_option[i])
                    current_option[i].num_goal_hits += 1

                if rd or option_terminated or option_duration[i] >= 1000:
                    if len(option_trajectories[i]) > 0 and not current_option[i].is_global_option:
                        print("option update", current_option[i], "traj len:", len(option_trajectories[i]), option_terminated, "|", rd, option_duration[i] >= 1000)
                        traj = [trajectories[i][j] for j in option_trajectories[i]]
                        traj = ((np.stack(traj) - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
                        current_option[i].derive_positive_and_negative_examples(traj, option_terminated)
                        current_option[i].fit_initiation_classifier()
                    option_trajectories[i].clear()
                    option_duration[i] = 0
                    current_option[i] = None
                else:
                    option_duration[i] += 1

                if rd:
                    traj = ((np.stack(trajectories[i]) - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
                    drn_model.train_rel_nov(traj)
                    trajectories[i].clear()

                trajectories[i].append(s)
                option_trajectories[i].append(len(trajectories[i])-1)

                next_states.append(s)
                # rewards.append(r)
                # dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(s)


            next_states = np.stack(next_states)

            # rewards = np.hstack(rewards)
            #dones = np.hstack(dones)
            # real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)


            # total reward = int reward + ext Reward
            # intrinsic_reward = agent.compute_intrinsic_reward(
            #     ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
            # intrinsic_reward = np.hstack(intrinsic_reward)
            # sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs)
            # total_int_reward.append(intrinsic_reward)
            # total_state.append(states)
            # total_reward.append(rewards)
            # total_done.append(dones)
            # total_action.append(actions)
            # total_ext_values.append(value_ext)
            # total_int_values.append(value_int)
            # total_policy.append(policy)
            # total_policy_np.append(policy.cpu().numpy())
            states = next_states
            del next_states

            # sample_rall += log_rewards[sample_env_idx]
            #
            # sample_step += 1
            # if real_dones[sample_env_idx]:
            #     sample_episode += 1
            #     writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
            #     writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
            #     writer.add_scalar('data/step', sample_step, sample_episode)
            #     sample_rall = 0
            #     sample_step = 0
            #     sample_i_rall = 0


            # writer.add_scalar('data/avg_reward_per_step', np.mean(rewards), global_step + num_worker * (cur_step - num_step))

        # while all(episode_rewards):
        #     global_ep += 1
        #     avg_ep_reward = np.mean([env_ep_rewards.pop(0) for env_ep_rewards in episode_rewards])
        #     writer.add_scalar('data/avg_reward_per_episode', avg_ep_reward, global_ep)

        # _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        # total_ext_values.append(value_ext)
        # total_int_values.append(value_int)
        # # --------------------------------------------------
        #
        # total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        # total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        # total_action = np.stack(total_action).transpose().reshape([-1])
        # total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        # total_ext_values = np.stack(total_ext_values).transpose()
        # total_int_values = np.stack(total_int_values).transpose()
        # total_logging_policy = np.vstack(total_policy_np)
        #
        # # Step 2. calculate intrinsic reward
        # # running mean intrinsic reward
        # total_int_reward = np.stack(total_int_reward).transpose()
        # #total_int_reward = np.stack(total_int_reward).swapaxes(0, 1)
        # total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
        #                                  total_int_reward.T])
        # mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        # reward_rms.update_from_moments(mean, std ** 2, count)
        #
        # # normalize intrinsic reward
        # total_int_reward /= np.sqrt(reward_rms.var)
        # writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        # writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # # -------------------------------------------------------------------------------------------
        #
        # # logging Max action probability
        # writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)
        #
        # # Step 3. make target and advantage
        # # extrinsic reward calculate
        # ext_target, ext_adv = make_train_data(total_reward,
        #                                       total_done,
        #                                       total_ext_values,
        #                                       gamma,
        #                                       num_step,
        #                                       num_worker)
        #
        # # intrinsic reward calculate
        # # None Episodic
        # int_target, int_adv = make_train_data(total_int_reward,
        #                                       np.zeros_like(total_int_reward),
        #                                       total_int_values,
        #                                       int_gamma,
        #                                       num_step,
        #                                       num_worker)
        #
        # # add ext adv and int adv
        # total_adv = int_adv * int_coef + ext_adv * ext_coef
        # # -----------------------------------------------
        #
        # # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        option_handler.create_new_option_if_possible()

        # -----------------------------------------------

        # Step 5. Training!
        # agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
        #                   total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
        #                   total_policy)

        # if global_step % (num_worker * num_step * 100) == 0:
        #     print('Now Global Step :{}'.format(global_step))
        #     torch.save(agent.model.state_dict(), model_path)
        #     torch.save(agent.rnd.predictor.state_dict(), predictor_path)
        #     torch.save(agent.rnd.target.state_dict(), target_path)\

        del rewards
        del dones
        del real_dones
        del log_rewards
        del next_obs

        del total_state
        del total_reward
        del total_done
        del total_action
        del total_int_reward
        del total_next_obs
        del total_ext_values
        del total_int_values
        del total_policy
        del total_policy_np


if __name__ == '__main__':
    main()
