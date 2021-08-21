
"""

List of parallel envs
At each step each env returns (s, a, s', r, d, op) tuple

train checks if option hits termination and acts on env with new option
option handler used to select new options from states. Also holds global option

train feeds tuples to Global PPO-RND (primative action option)
train adds tuples to respective option buffers for Option policy updates.
train feeds tuples to Policy over options (optional for now)

"""

import numpy as np
import matplotlib.pyplot as plt

from agent.OptionClass import Option
from sklearn.svm import OneClassSVM, SVC


class OptionHandler:

    def __init__(self, drn_model, state_dim, action_dim, device, buffer_length=100, gestation_period=25, max_steps=500):

        self.drn_model = drn_model

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lr_c = 0
        self.lr_a = 0

        self.device = device

        self.buffer_length = buffer_length
        self.gestation_period = gestation_period
        self.max_steps = max_steps

        self.clear_option_buffers = False

        self.options = []
        self.new_options = []
        self.mature_options = []

        self.update_buffers = {} #dict from option to list

        global_option = self.create_new_option("", is_global=True)
        self.global_option = global_option

        self.new_options.append(global_option)
        self.options.append(global_option)


    def update(self, state, action, reward, next_state, done, op):
        # if op not in self.update_buffers.keys():
        #     self.update_buffers[op] = []
        # #TODO extract features from states
        # self.update_buffers[op].append((state, action, reward, next_state, done))
        #
        # if len(self.update_buffers[op]) >= self.batch_size:
        #     self.train_option(op, self.update_buffers[op])
        op.update_model(state, action, reward, next_state, done)

        if done:
            self.manage_option_after_execution(op)


    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return True
        return False

    def create_new_option(self, name, is_global=False):
        if is_global:
            option = Option(termination_set=None, state_dim=self.state_dim, action_dim=self.action_dim, buffer_length=self.buffer_length,
                                      global_init=True,
                                      gestation_period=self.gestation_period,
                                      timeout=200, max_steps=self.max_steps, device=self.device,
                                      name="global-option",
                                      option_idx=len(self.options),
                                      lr_c=self.lr_c, lr_a=self.lr_a, global_option=None)
        else:
            traj = self.drn_model.last_trajectory

            _, _, _, freq_vals, I = self.drn_model.get_latest_subgoals()

            if len(I) == 0:
                #option creation failed, no salient event
                return None

            event_I = I[np.argmax(freq_vals)]
            window = 5
            states = traj[event_I-window : event_I + window]

            option = Option(termination_set=states, state_dim=self.state_dim, action_dim=self.action_dim,
                                      buffer_length=self.buffer_length,
                                      global_init=True,
                                      gestation_period=self.gestation_period,
                                      timeout=200, max_steps=self.max_steps, device=self.device,
                                      name=name,
                                      option_idx=len(self.options),
                                      lr_c=self.lr_c, lr_a=self.lr_a, global_option=self.global_option)




            S = np.mean(states, axis=(0,1))
            plt.title("termination set")
            plt.imshow(S)
            plt.savefig("option_plots_test/term_" + name)

            # target_salient_event = option.create_termination_classifier(states)
            # option.set_target_salient_event(target_salient_event)

            option.derive_positive_and_negative_examples(traj)
            option.fit_initiation_classifier()
        return option


    def clear_replay_buffer(self, option):
        assert isinstance(option, ModelBasedOption)
        option.value_learner.replay_buffer.clear()

    def create_new_option_if_possible(self):
        #make sure that drn is trained and meaningful subgoal can be found
        if self.drn_model.is_drn_trained() and self.should_create_new_option():
            #ask the Deep Rel Nov to make a subgoal from the latest traj or pass in traj
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_new_option(name)
            if new_option is None:
                return #option could not be made
            self.new_options.append(new_option)
            self.options.append(new_option)

    def manage_option_after_execution(self, executed_option):
        #If option finished gestation add it to mature option list
        if executed_option in self.new_options and not executed_option.is_in_training_phase():
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

            if self.clear_option_buffers:
                self.clear_replay_buffer(executed_option)

        #clear option buffer every 2 * executed_option.gestation_period goal hits.
        if executed_option.num_goal_hits >= 2 * executed_option.gestation_period and self.clear_option_buffers:
            self.clear_replay_buffer(executed_option)

        if not executed_option.is_global_option:
            executed_option.update_termination_classifier()
