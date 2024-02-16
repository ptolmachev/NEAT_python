import numpy as np
import gymnasium
from gymnasium import spaces

class CDDM_environment(gymnasium.Env):
    def __init__(self, n_steps=300, seed=None):
        super(CDDM_environment, self).__init__()
        self.n_outputs = 2
        self.n_inputs = 6
        self.n_steps = n_steps
        self.t = 0
        if seed is None:
            seed = np.random.randint(10000)
        else:
            seed = seed
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(self.n_outputs)
        self.observation_space = spaces.Box(low=np.zeros(self.n_inputs),
                                            high=np.ones(self.n_inputs),
                                            shape=(self.n_inputs,))


    def set_inputs_and_targets(self):
        contex_ind = 0 if self.rng.random() > 0.5 else 1
        input_irrelevant_right = self.rng.random()
        input_irrelevant_left = 1.0 - input_irrelevant_right
        input_relevant_right = self.rng.random()
        input_relevant_left = 1.0 - input_relevant_right
        self.inputs = np.zeros(self.n_inputs)
        self.inputs[contex_ind] = 1.0
        if contex_ind == 0:
            self.inputs[2:] = np.array([input_relevant_right, input_relevant_left,
                                        input_irrelevant_right, input_irrelevant_left])
        else:
            self.inputs[2:] = np.array([input_irrelevant_right, input_irrelevant_left,
                                        input_relevant_right, input_relevant_left])
        self.target = np.zeros(2)
        target_ind = 0 if input_relevant_right > input_relevant_left else 1
        self.target[target_ind] = 1.0
        return self.inputs, self.target

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self, options = None, seed=None):
        if not (seed is None):
            self.seed(seed)
        self.t = 0
        inputs, target = self.set_inputs_and_targets()
        return inputs, {"inputs" : inputs, "target" : target}

    def step(self, output):
        done = False
        reward = -np.sum((self.target - output) ** 2)
        self.t += 1
        if self.t == self.n_steps:
            done = True
        return self.inputs, reward, done, done, {"inputs": self.inputs, "targets": self.target}

