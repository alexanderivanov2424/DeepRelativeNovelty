import torch
import random
import itertools
import numpy as np
from scipy.spatial import distance
from thundersvm import OneClassSVM, SVC
# from sklearn.svm import OneClassSVM, SVC

from agent.dynamics.mpc import MPC

from agent.ppo.PPOAgentClass import PPOAgent

from collections import deque


class Option(object):
    def __init__(self, *, name, termination_set, state_dim, action_dim, buffer_length, global_init, gestation_period,
                 timeout, max_steps, device, option_idx, lr_c, lr_a, global_option,
                 path_to_model=None):
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
        self.global_init = global_init #is global option
        self.buffer_length = buffer_length

        self.global_option = global_option
        self.is_global_option = global_option is None

        self.seed = 0
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period

        self.positive_examples = deque(maxlen=500) #100 * 3136 = 313600
        self.negative_examples = deque(maxlen=1)
        self.optimistic_classifier = None
        self.pessimistic_classifier = None

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.solver = PPOAgent(obs_n_channels=self.state_dim + 1, n_actions=self.action_dim, device_id=-1)

        self.termination_set = termination_set
        self.update_termination_classifier()

        self.children = []
        self.success_curve = []
        self.effect_set = []

        if path_to_model:
            print(f"Loading model from {path_to_model} for {self.name}")
            self.solver.load_model(path_to_model)

        # if self.use_vf and not self.use_global_vf and self.parent is not None:
        #     self.initialize_value_function_with_global_value_function()

        print(f"Created model-based option {self.name} with option_idx={self.option_idx}")

    # ------------------------------------------------------------
    # Learning Phase Methods
    # ------------------------------------------------------------

    def get_training_phase(self):
        if self.num_goal_hits < self.gestation_period:
            return "gestation"
        return "initiation_done"

    def is_in_training_phase(self):
        return self.num_goal_hits < self.gestation_period

    def extract_state_features(self, state):
        # features = state if isinstance(state, np.ndarray) else state.features()
        # if "push" in self.mdp.env_name:
        #     return features[:4]
        # return features[:2]
        return state

    def is_init_true(self, state):
        if self.is_global_option:
            return True

        if self.global_init or self.get_training_phase() == "gestation":
            return True

        # if self.is_last_option and self.mdp.get_start_state_salient_event()(state):
        #     return True

        features = self.extract_state_features(state)
        return self.optimistic_classifier.predict([features])[0] == 1 or self.pessimistic_is_init_true(state)

    def is_in_term_set(self, state):
        state = np.reshape(state, (4,84,84))
        features = np.reshape(self.global_option.solver.get_features(state), (1, -1))
        return self.termination_classifier.predict(features)[0] == 1

    def is_term_true(self, state):
        if self.is_global_option:
            self.steps = 0
            return True
        if self.steps > self.max_steps:
            self.steps = 0
            return True
        return self.is_in_term_set(state)

    def pessimistic_is_init_true(self, state):
        if self.is_global_option:
            return True

        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.extract_state_features(state)
        return self.pessimistic_classifier.predict([features])[0] == 1


    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def _get_epsilon(self):
        if self.num_goal_hits <= 3:
            return 0.8
        return 0.2

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


    # ------------------------------------------------------------
    # Hindsight Experience Replay
    # ------------------------------------------------------------

    def update_value_function(self, option_transitions, reached_goal, pursued_goal):
        """ Update the goal-conditioned option value function. """

        self.experience_replay(option_transitions, pursued_goal)
        self.experience_replay(option_transitions, reached_goal)

    def initialize_value_function_with_global_value_function(self):
        self.value_learner.actor.load_state_dict(self.global_value_learner.actor.state_dict())
        self.value_learner.critic.load_state_dict(self.global_value_learner.critic.state_dict())
        self.value_learner.target_actor.load_state_dict(self.global_value_learner.target_actor.state_dict())
        self.value_learner.target_critic.load_state_dict(self.global_value_learner.target_critic.state_dict())

    def experience_replay(self, trajectory, goal_state):
        for state, action, reward, next_state in trajectory:
            augmented_state = self.get_augmented_state(state, goal=goal_state)
            augmented_next_state = self.get_augmented_state(next_state, goal=goal_state)
            done = self.is_at_local_goal(next_state, goal_state)

            reward_func = self.overall_mdp.dense_gc_reward_function if self.dense_reward \
                else self.overall_mdp.sparse_gc_reward_function
            reward, global_done = reward_func(next_state, goal_state, info={})

            if not self.use_global_vf or self.global_init:
                self.value_learner.step(augmented_state, action, reward, augmented_next_state, done)

            # Off-policy updates to the global option value function
            if not self.global_init:
                assert self.global_value_learner is not None
                self.global_value_learner.step(augmented_state, action, reward, augmented_next_state, global_done)

    def value_function(self, states, goals):
        assert isinstance(states, np.ndarray)
        assert isinstance(goals, np.ndarray)

        if len(states.shape) == 1:
            states = states[None, ...]
        if len(goals.shape) == 1:
            goals = goals[None, ...]

        goal_positions = goals[:, :2]
        augmented_states = np.concatenate((states, goal_positions), axis=1)
        augmented_states = torch.as_tensor(augmented_states).float().to(self.device)

        if self.use_global_vf and not self.global_init:
            values = self.global_value_learner.get_values(augmented_states)
        else:
            values = self.value_learner.get_values(augmented_states)

        return values

    # ------------------------------------------------------------
    # Learning Initiation Classifiers
    # ------------------------------------------------------------

    def get_first_state_in_classifier(self, trajectory, classifier_type="pessimistic"):
        """ Extract the first state in the trajectory that is inside the initiation classifier. """

        assert classifier_type in ("pessimistic", "optimistic"), classifier_type
        classifier = self.pessimistic_is_init_true if classifier_type == "pessimistic" else self.is_init_true
        for state in trajectory:
            if classifier(state):
                return state
        return None

    def sample_from_initiation_region_fast(self):
        """ Sample from the pessimistic initiation classifier. """
        num_tries = 0
        sampled_state = None
        while sampled_state is None and num_tries < 200:
            num_tries = num_tries + 1
            sampled_trajectory_idx = random.choice(range(len(self.positive_examples)))
            sampled_trajectory = self.positive_examples[sampled_trajectory_idx]
            sampled_state = self.get_first_state_in_classifier(sampled_trajectory)
        return sampled_state

    def sample_from_initiation_region_fast_and_epsilon(self):
        """ Sample from the pessimistic initiation classifier. """
        def compile_states(s):
            pos0 = self.mdp.get_position(s)
            pos1 = np.copy(pos0)
            pos1[0] -= self.target_salient_event.tolerance
            pos2 = np.copy(pos0)
            pos2[0] += self.target_salient_event.tolerance
            pos3 = np.copy(pos0)
            pos3[1] -= self.target_salient_event.tolerance
            pos4 = np.copy(pos0)
            pos4[1] += self.target_salient_event.tolerance
            return pos0, pos1, pos2, pos3, pos4

        idxs = [i for i in range(len(self.positive_examples))]
        random.shuffle(idxs)

        for idx in idxs:
            sampled_trajectory = self.positive_examples[idx]
            states = []
            for s in sampled_trajectory:
                states.extend(compile_states(s))

            position_matrix = np.vstack(states)
            # optimistic_predictions = self.optimistic_classifier.predict(position_matrix) == 1
            # pessimistic_predictions = self.pessimistic_classifier.predict(position_matrix) == 1
            # predictions = np.logical_or(optimistic_predictions, pessimistic_predictions)
            predictions = self.pessimistic_classifier.predict(position_matrix) == 1
            predictions = np.reshape(predictions, (-1, 5))
            valid = np.all(predictions, axis=1)
            indices = np.argwhere(valid == True)
            if len(indices) > 0:
                return sampled_trajectory[indices[0][0]]

        return self.sample_from_initiation_region_fast()

    def derive_positive_and_negative_examples(self, visited_states, success=None):
        start_state = visited_states[0]
        final_state = visited_states[-1]

        if success is None:
            success = self.is_term_true(final_state)

        if success:
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            self.positive_examples.extend(positive_states)
        else:
            negative_examples = [start_state]
            self.negative_examples.extend(negative_examples)

        import matplotlib.pyplot as plt
        arr = np.array(self.positive_examples)
        if arr.shape[0] > 0:
            S = np.mean(arr, axis=(0,1))
            plt.title("initation set positive examples")
            plt.imshow(S)
            plt.savefig("option_plots_test/positive_" + self.name + "_")

        arr = np.array(self.negative_examples)
        if arr.shape[0] > 0:
            S = np.mean(arr, axis=(0,1))
            plt.title("initation set negative examples")
            plt.imshow(S)
            plt.savefig("option_plots_test/negative_" + self.name + "_")


    def update_termination_classifier(self, nu=0.1):
        if not self.termination_set is None:
            features = [self.global_option.solver.get_features(obs) for obs in self.termination_set]
            positive_feature_matrix = self.construct_feature_matrix(features)
            self.termination_classifier = OneClassSVM(kernel="rbf", nu=nu)
            self.termination_classifier.fit(positive_feature_matrix)

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    def construct_feature_matrix(self, examples):
        # states = list(itertools.chain.from_iterable(examples))
        positions = [np.reshape(self.extract_state_features(state), (-1,)) for state in examples]
        return np.array(positions)

    def train_one_class_svm(self, nu=0.1):  # TODO: Implement gamma="auto" for thundersvm
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu)
        self.pessimistic_classifier.fit(positive_feature_matrix)

        self.optimistic_classifier = OneClassSVM(kernel="rbf", nu=nu/10.)
        self.optimistic_classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)

        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        if negative_feature_matrix.shape[0] >= 10:  # TODO: Implement gamma="auto" for thundersvm
            kwargs = {"kernel": "rbf", "gamma": "auto", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "auto"}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu)
            self.pessimistic_classifier.fit(positive_training_examples)

    # ------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------

    def get_states_inside_pessimistic_classifier_region(self):
        point_array = self.construct_feature_matrix(self.positive_examples)
        point_array_predictions = self.pessimistic_classifier.predict(point_array)
        positive_point_array = point_array[point_array_predictions == 1]
        return positive_point_array

    def distance_to_state(self, state, metric="euclidean"):
        """ Compute the distance between the current option and the input `state`. """

        assert metric in ("euclidean", "value"), metric
        if metric == "euclidean":
            return self._euclidean_distance_to_state(state)
        return self._value_distance_to_state(state)

    def _euclidean_distance_to_state(self, state):
        point = self.mdp.get_position(state)

        assert isinstance(point, np.ndarray)
        assert point.shape == (2,), point.shape

        positive_point_array = self.get_states_inside_pessimistic_classifier_region()

        distances = distance.cdist(point[None, :], positive_point_array)
        return np.median(distances)

    def _value_distance_to_state(self, state):
        features = state.features() if not isinstance(state, np.ndarray) else state
        goals = self.get_states_inside_pessimistic_classifier_region()

        distances = self.value_function(features, goals)
        distances[distances > 0] = 0.
        return np.median(np.abs(distances))

    # ------------------------------------------------------------
    # Convenience functions
    # ------------------------------------------------------------

    def get_option_success_rate(self):
        if self.num_executions > 0:
            return self.num_goal_hits / self.num_executions
        return 1.

    def get_success_rate(self):
        if len(self.success_curve) == 0:
            return 0.
        return np.mean(self.success_curve)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Option):
            return self.name == other.name
        return False
