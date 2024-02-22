import numpy as np
from copy import deepcopy
from sklearn.datasets import make_circles, make_moons
'''
Generic class-template for task which should contain 'generate_input_target_stream' and 'get_batch' methods
'''
class TaskXOR():
    def __init__(self):
        self.n_inputs = 3
        self.n_outputs = 1
        self.observation_space = np.ones(self.n_inputs)
        self.action_space = np.ones(self.n_outputs)

    def get_batch(self, batch_size=None, seed=None):
        '''
        '''
        if seed is None:
            rng = np.random.default_rng(np.random.randint(10000))
        else:
            rng = np.random.default_rng(seed)

        # Generate random values for v1 and v2
        v1 = 2 * rng.uniform(size=(batch_size, )) - 1
        v2 = 2 * rng.uniform(size=(batch_size, )) - 1

        # Initialize inputs and targets arrays
        inputs = np.zeros((self.n_inputs, batch_size))
        targets = np.zeros((self.n_outputs, batch_size))

        # Assign values to inputs
        inputs[0, :] = v1
        inputs[1, :] = v2
        inputs[2, :] = 1.0  # Constantly active input

        # Conditionally assign values to targets based on the sign of v1 and v2
        mask = np.sign(v1) == np.sign(v2)
        targets[0, mask] = 1.0
        # targets[1, ~mask] = 1.0
        return inputs, targets


class TaskCDDM():

    def __init__(self):
        self.n_inputs = 6
        self.n_outputs = 2
        self.observation_space = np.ones(self.n_inputs)
        self.action_space = np.ones(self.n_outputs)


    def get_batch(self, batch_size=256, seed=None):
        if seed is None:
            rng = np.random.default_rng(np.random.randint(10000))
        else:
            rng = np.random.default_rng(seed)
        inputs = np.zeros((self.n_inputs, batch_size))
        targets = np.zeros((self.n_outputs, batch_size))
        for context_ind in [0, 1]:
            for i in range(batch_size//2):
                index = context_ind * batch_size// 2 + i
                relevant_right = rng.uniform()
                relevant_left = 1 - relevant_right
                irrelevant_right = rng.uniform()
                irrelevant_left = 1 - irrelevant_right
                rel = [relevant_right, relevant_left]
                irr = [irrelevant_right, irrelevant_left]
                correct_choice_ind = 1 if relevant_left > relevant_right else 0
                inputs[context_ind, index] = 1.0
                if context_ind == 0:
                    inputs[2:, index] = [*rel, *irr]
                else:
                    inputs[2:, index] = [*irr, *rel]
                targets[correct_choice_ind, index] = 1.0
        return inputs, targets


class TaskMoons():
    def __init__(self):
        self.n_inputs = 3
        self.n_outputs = 1
        self.observation_space = np.ones(self.n_inputs)
        self.action_space = np.ones(self.n_outputs)

    def get_batch(self, batch_size=None, seed=None):
        '''
        '''
        if seed is None:
            seed = np.random.randint(10000)
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(seed)

        X, targets = make_moons(n_samples=batch_size, noise=0.2, random_state=seed)
        inputs = np.zeros((self.n_inputs, batch_size))
        inputs[2, :] = 1.0
        inputs[:2, :] = X.T
        return inputs, targets

class TaskCircles():
    def __init__(self):
        self.n_inputs = 3
        self.n_outputs = 1
        self.observation_space = np.ones(self.n_inputs)
        self.action_space = np.ones(self.n_outputs)

    def get_batch(self, batch_size=None, seed=None):
        '''
        '''
        if seed is None:
            seed = np.random.randint(10000)
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(seed)

        X, targets = make_circles(n_samples=batch_size, noise=0.1, random_state=seed)
        inputs = np.zeros((self.n_inputs, batch_size))
        inputs[2, :] = 1.0
        inputs[:2, :] = X.T
        return inputs, targets



class TaskSpirals():
    def __init__(self):
        self.n_inputs = 3
        self.n_outputs = 1
        self.observation_space = np.ones(self.n_inputs)
        self.action_space = np.ones(self.n_outputs)
        self.L = 1
        self.w = 3 * np.pi


    def make_spirals(self, n_samples, noise, seed):
        t = self.L * np.random.rand(n_samples)
        rng = np.random.default_rng(seed)
        ids = rng.choice([0, 1], size=n_samples)
        mask = ids == 0

        psi = np.zeros(n_samples)
        psi[mask] = 0
        psi[~mask] = -np.pi

        r = t + noise * rng.normal(size=n_samples)
        phi = self.w * t + psi
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.vstack([x.reshape(1, -1), y.reshape(1, -1)]), mask

    def get_batch(self, batch_size=None, seed=None):
        X, targets = self.make_spirals(n_samples=batch_size, noise=0.05, seed=seed)
        inputs = np.zeros((self.n_inputs, batch_size))
        inputs[2, :] = 1.0
        inputs[:2, :] = X
        return inputs, targets


