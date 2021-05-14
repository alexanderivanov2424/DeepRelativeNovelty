from rnd.model import RNDModel

class DeepRelNov:
    def __init__(self, novelty_rnd, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.n_l = 7
        self.novelty_rnd = novelty_rnd
        self.novelty_threshold = .15
        self.frequency_rnd = RNDModel(input_size, output_size)
        self.frequency_threshold = .68

    def compute_rnd_output(self, obs, rnd_model):
        obs = torch.FloatTensor(obs).to(self.device)

        target_next_feature = rnd.target(obs)
        predict_next_feature = rnd.predictor(obs)
        output = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return output.data.cpu().numpy()

    def get_subgoals(self, trajectory):
        novelty_vals = trajectory.map(lambda x: self.compute_rnd_output(x, self.novelty_rnd))

	def get_rel_novelty(i):
	    visits_before = novelty_vals[i - n_l : i]
	    visits_after = novelty_vals[i : i + n_l]

	    return np.sqrt(np.sum(visits_before) / np.sum(visits_after))

        subgoals = []
        for i in range(n_l, len(trajectory) - n_l):
            rel_nov = get_rel_novelty(i)
            obs = trajectory[i]
            if rel_nov > self.novelty_threshold:
                freq_val = self.compute_rnd_output(obs, self.frequency_rnd)
                if freq_val < self.frequency_threshold:
                    subgoals.append(obs)
        return subgoals
