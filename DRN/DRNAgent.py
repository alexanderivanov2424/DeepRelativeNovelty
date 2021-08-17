import os
import time
import torch
import pickle
import argparse
import numpy as np
from copy import deepcopy
from functools import reduce
from collections import deque

from model import RNDModel


from agent.OptionClass import Option


class DRNAgent(object):
    def __init__(self, rnd_input_size, rnd_output_size):
        self.rnd = RNDModel(rnd_input_size, rnd_output_size)

    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.tensor(next_obs, dtype=torch.float)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    @staticmethod
    def _pick_earliest_option(state, options):
        for option in options:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option

    def act(self, state, option_handler):
        # current_option = self._pick_earliest_option(state, self.chain)
        # return current_option if current_option is not None else self.global_option
        available_options = []
        for option in option_handler.options:
            # if option == option_handler.global_option:
            #     continue
            if option.is_init_true(state):
                available_options.append(option)
        if len(available_options) == 0:
            return option_handler.global_option
        return np.random.choice(available_options)

    def get_action(self, states):
        return [self.act(state) for state in states]

    def random_rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_state.is_terminal():
            state = deepcopy(self.mdp.cur_state)
            action = self.mdp.sample_random_action()
            reward, next_state = self.mdp.execute_agent_action(action)
            # self.global_option.update_model(state, action, reward, next_state)
            step_number += 1
        return step_number

    def rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_state.is_terminal():
            state = deepcopy(self.mdp.cur_state)

            selected_option, subgoal = self.act(state)

            # Overwrite the subgoal for the global-option
            # if selected_option == self.global_option and self.use_global_option_subgoals:
            #     subgoal = self.pick_subgoal_for_global_option(state)
            if selected_option is None:
                action = self.mdp.sample_random_action()
                reward, next_state = self.mdp.execute_agent_action(action)
                step_number += 1
                continue

            transitions, reward = selected_option.rollout(step_number=step_number, rollout_goal=subgoal)

            if len(transitions) == 0:
                break

            self.manage_option_after_rollout(selected_option)
            step_number += len(transitions)
        return step_number

    def run_loop(self, num_episodes, num_steps, start_episode=0):
        #TODO need to be changed

        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(start_episode, start_episode + num_episodes):
            self.reset(episode)

            step = self.dsc_rollout(num_steps) if episode > self.warmup_episodes else self.random_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            if episode == self.warmup_episodes - 1 and self.use_model:
                self.learn_dynamics_model(epochs=50)
            elif episode >= self.warmup_episodes and self.use_model:
                self.learn_dynamics_model(epochs=5)

            self.log_success_metrics(episode)

        return per_episode_durations

    def log_success_metrics(self, episode):
        individual_option_data = {option.name: option.get_option_success_rate() for option in self.chain}
        overall_success = reduce(lambda x,y: x*y, individual_option_data.values())
        self.log[episode] = {"individual_option_data": individual_option_data, "success_rate": overall_success}

        if episode % self.evaluation_freq == 0 and episode > self.warmup_episodes:
            success, step_count = test_agent(self, 1, self.max_steps)

            self.log[episode]["success"] = success
            self.log[episode]["step-count"] = step_count[0]

            with open(f"{self.experiment_name}/log_file_{self.seed}.pkl", "wb+") as log_file:
                pickle.dump(self.log, log_file)

    def learn_dynamics_model(self, epochs=50, batch_size=1024):
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=epochs, batch_size=batch_size)
        for option in self.chain:
            option.solver.model = self.global_option.solver.model

    def is_chain_complete(self):
        return all([option.get_training_phase() == "initiation_done" for option in self.chain]) and self.mature_options[-1].is_init_true(np.array([0,0]))

    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and \
                not self.contains_init_state()
        return False

    def contains_init_state(self):
        for option in self.mature_options:
            if option.is_init_true(np.array([0,0])):  # TODO: Get test-time start state automatically
                return True
        return False

    def manage_option_after_rollout(self, executed_option):
        #TODO Don't have a chain anymore
        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

            if self.clear_option_buffers:
                self.filter_replay_buffer(executed_option)

        if executed_option.num_goal_hits == 2 * executed_option.gestation_period and self.clear_option_buffers:
            self.filter_replay_buffer(executed_option)

        if self.should_create_new_option():
            #ask the Deep Rel Nov to make a subgoal from the latest traj or pass in traj
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_model_based_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.options.append(new_option)

    def find_nearest_option_in_chain(self, state):
        if len(self.mature_options) > 0:
            distances = [(option, option.distance_to_state(state)) for option in self.mature_options]
            nearest_option = sorted(distances, key=lambda x: x[1])[0][0]  # type: ModelBasedOption
            return nearest_option

    def pick_subgoal_for_global_option(self, state):
        nearest_option = self.find_nearest_option_in_chain(state)
        if nearest_option is not None:
            return nearest_option.sample_from_initiation_region_fast_and_epsilon()
        return self.global_option.get_goal_for_rollout()

    def filter_replay_buffer(self, option):
        assert isinstance(option, ModelBasedOption)
        print(f"Clearing the replay buffer for {option.name}")
        option.value_learner.replay_buffer.clear()

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")

        if episode % self.logging_freq == 0 and episode != 0:
            options = self.mature_options + self.new_options

            for option in self.mature_options:
                episode_label = episode if self.generate_init_gif else -1
                plot_two_class_classifier(option, episode_label, self.experiment_name, plot_examples=True)

            for option in options:
                if self.use_global_vf:
                    make_chunked_goal_conditioned_value_function_plot(option.global_value_learner,
                                                                    goal=option.get_goal_for_rollout(),
                                                                    episode=episode, seed=self.seed,
                                                                    experiment_name=self.experiment_name,
                                                                    option_idx=option.option_idx)
                else:
                    make_chunked_goal_conditioned_value_function_plot(option.value_learner,
                                                                    goal=option.get_goal_for_rollout(),
                                                                    episode=episode, seed=self.seed,
                                                                    experiment_name=self.experiment_name)

    # We aren't building chains so never need to do this.
    # Only want global options

    # def create_model_based_option(self, name, parent=None):
    #     option_idx = len(self.chain) + 1 if parent is not None else 1
    #     option = ModelBasedOption(parent=parent, mdp=self.mdp,
    #                               buffer_length=self.buffer_length,
    #                               global_init=False,
    #                               gestation_period=self.gestation_period,
    #                               timeout=200, max_steps=self.max_steps, device=self.device,
    #                               target_salient_event=self.target_salient_event,
    #                               name=name,
    #                               path_to_model="",
    #                               global_solver=self.global_option.solver,
    #                               use_vf=self.use_vf,
    #                               use_global_vf=self.use_global_vf,
    #                               use_model=self.use_model,
    #                               dense_reward=self.use_dense_rewards,
    #                               global_value_learner=self.global_option.value_learner,
    #                               option_idx=option_idx,
    #                               lr_c=self.lr_c, lr_a=self.lr_a,
    #                               multithread_mpc=self.multithread_mpc)
    #     return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=True,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_global_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=None,
                                  option_idx=0,
                                  lr_c=self.lr_c, lr_a=self.lr_a,
                                  multithread_mpc=self.multithread_mpc)
        return option

    def reset(self, episode):
        self.mdp.reset()

        if self.use_diverse_starts and episode > self.warmup_episodes:
            random_state = self.mdp.sample_random_state()
            random_position = self.mdp.get_position(random_state)
            self.mdp.set_xy(random_position)

def test_agent(exp, num_experiments, num_steps):
    def rollout():
        step_number = 0
        while step_number < num_steps and not exp.mdp.sparse_gc_reward_function(exp.mdp.cur_state, exp.mdp.goal_state, {})[1]:

            state = deepcopy(exp.mdp.cur_state)
            selected_option, subgoal = exp.act(state)
            transitions, reward = selected_option.rollout(step_number=step_number, rollout_goal=subgoal, eval_mode=True)
            step_number += len(transitions)
        return step_number

    success = 0
    step_counts = []

    for _ in tqdm(range(num_experiments), desc="Performing test rollout"):
        exp.mdp.reset()
        steps_taken = rollout()
        if steps_taken != num_steps:
            success += 1
        step_counts.append(steps_taken)

    print("*" * 80)
    print(f"Test Rollout Success Rate: {success / num_experiments}, Duration: {np.mean(step_counts)}")
    print("*" * 80)

    return success / num_experiments, step_counts

def get_trajectory(exp, num_steps):
    exp.mdp.reset()

    traj = []
    step_number = 0

    while step_number < num_steps and not exp.mdp.sparse_gc_reward_function(exp.mdp.cur_state, exp.mdp.goal_state, {})[1]:
        state = deepcopy(exp.mdp.cur_state)
        selected_option, subgoal = exp.act(state)
        transitions, reward = selected_option.rollout(step_number=step_number, rollout_goal=subgoal, eval_mode=True)
        step_number += len(transitions)
        traj.append((selected_option.name, transitions))
    return traj, step_number
