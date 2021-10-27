from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from agents import *
from config import *
from envs import *
from utils import *

from deviation_subgoal_generator import DeviationSubgoalGenerator
from options.GlobalOption import GlobalOption

import csv


def main():
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    assert train_method == 'RND'
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']


    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    run_path = Path(f'runs/{env_id}_{datetime.now().strftime("%b%d_%H-%M-%S")}')

    run_path.mkdir(parents=True)

    with open(run_path / 'step_data.csv','w+') as fd:
        #env_num, ep num, num option executions total, num actions executions total, ext op reward, done, real_done, action, player_pos, int_reward_per_one_decision
        csv_writer = csv.writer(fd, delimiter=',')
        csv_writer.writerow(['environment number','episode number', 'total action executions', 'extrinsic option reward', 'done', 'real done', 'action', 'player position (x,y)', 'intrinsic reward for one decision'])

    with open(run_path / 'episode_data.csv','w+') as fd:
        #env_num, ep num, ep_rew, number of options, number of actions, int_reward_per_epi
        csv_writer = csv.writer(fd, delimiter=',')
        csv_writer.writerow(['environment number','episode number', 'episode reward', 'episode length', 'intrinsic reward per episode'])


    global_option = GlobalOption(default_config, run_path)

    dev_subgoal_gen = DeviationSubgoalGenerator(global_option.agent.rnd)

    use_cuda = default_config.getboolean('UseGPU')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if use_cuda else 'torch.FloatTensor')

    num_envs = int(default_config['NumEnv'])

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    num_step = int(default_config['NumStep'])
    pre_obs_norm_step = int(default_config['ObsNormStep'])


    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    else:
        raise NotImplementedError


    if is_load_model:
        print('load model...')
        if use_cuda:
            global_option.agent.model.load_state_dict(torch.load(model_path))
            global_option.agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            global_option.agent.rnd.target.load_state_dict(torch.load(target_path))
        else:
            global_option.agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            global_option.agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            global_option.agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
        print('load finished!')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_envs):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)


    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    for _ in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, global_option.output_size, size=(num_envs,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr, _ = parent_conn.recv()
            global_option.init_update(s)

    print('End to initalize...')

    total_action_executions = 0

    episode_counter = [0 for _ in range(num_envs)]
    episode_rewards = [0 for _ in range(num_envs)]
    episode_action_trajectories = [[] for _ in range(num_envs)]
    episode_length = [0 for _ in range(num_envs)]

    state_trajectories = [[] for _ in range(num_envs)]

    step_rewards = [[] for _ in range(num_envs)]

    global_step = 0
    global_ep = 0

    while True:
        global_step += 1

        actions, value_ext, value_int, policy = global_option.act()

        executed_actions = [None for _ in parent_conns]
        for i, (parent_conn, action) in enumerate(zip(parent_conns, actions)):
            parent_conn.send(action)
            episode_action_trajectories[i].append(action)
            executed_actions[i] = action

        next_states, rewards, dones, real_dones, next_obs = [], [], [], [], []
        for i, parent_conn in enumerate(parent_conns):
            s, r, d, rd, lr, info = parent_conn.recv()
            next_states.append(s)
            rewards.append(r)
            dones.append(d)
            real_dones.append(rd)
            next_obs.append(s[-1, :, :].reshape([1, 84, 84]))

            if d or rd:
                dev_subgoal_gen.train_gaussian(state_trajectories[i])

                if global_step > 100:
                    subgoal_idx = dev_subgoal_gen.generate_subgoal_index(np.array(state_trajectories[i]), global_option.obs_rms)
                    if subgoal_idx is not None:
                        print(subgoal_idx, len(state_trajectories[i]), subgoal_idx/len(state_trajectories[i]))

                state_trajectories[i] = []

            state_trajectories[i].append(s[-1, :, :].reshape([1, 84, 84]))

            episode_rewards[i] += r
            total_action_executions += 1

            with open(run_path / 'step_data.csv','a+') as fd:
                csv_writer = csv.writer(fd, delimiter=',')
                csv_writer.writerow([i, episode_counter[i], total_action_executions, r, d, rd, executed_actions[i],
                    str(info['player_pos']), global_option.get_intrinsic_reward(s)])
                fd.flush()

            if rd:
                with open(run_path / 'episode_data.csv','a+') as fd:
                    csv_writer = csv.writer(fd, delimiter=',')
                    csv_writer.writerow([i, episode_counter[i], episode_rewards[i], len(episode_action_trajectories[i]), episode_length[i]])
                episode_counter[i] += 1
                episode_rewards[i] = 0
                episode_action_trajectories[i] = []

        global_option.update(actions, next_states, next_obs, rewards, dones, real_dones, policy, value_ext, value_int)


if __name__ == '__main__':
    main()
