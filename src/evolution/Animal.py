import inspect
import re
from copy import deepcopy
import numpy as np
from src.evolution.Blueprint import *
from jax import numpy as jnp
import jax

class Animal():
    def __init__(self, blueprint, blueprint_params, neuron_type='relu', action_noise=0.03, action_type='Discrete', action_bounds=(-1, 1)):
        self.blueprint = deepcopy(blueprint)
        self.W = self.blueprint.get_connectivity_matrix()
        self.b = self.blueprint.get_biases()
        self.action_noise = action_noise
        self.n_inputs = self.blueprint.n_inputs
        self.n_outputs = self.blueprint.n_outputs
        self.neuron_type = neuron_type
        self.neural_activity = None
        self.blueprint_params = blueprint_params
        match neuron_type:
            case "sigmoid": self.activation = lambda x: 1.0 / (np.exp(x) + 1.0)
            case "relu": self.activation = lambda x: np.maximum(0, x)
            case "tanh": self.activation = lambda x: np.tanh(x)
        function_np_str = inspect.getsource(self.activation)  # getting the code as a string
        function_jnp_str = re.sub(r"np", "jnp", function_np_str)
        self.activation_jnp = eval(function_jnp_str.split("= ")[1])
        self.action_type = action_type
        self.action_bounds = action_bounds
        self.fitness = None
        self.species_id = None

        self.inp_nrns = set(self.blueprint.get_neurons_by_type('i'))
        self.out_nrns = set(self.blueprint.get_neurons_by_type('o'))
        self.nrn_names = list(self.blueprint.genome_dict["neurons"].keys())

        # Get indices of input and output neurons
        self.inp_nrns_inds = [self.nrn_names.index(nrn) for nrn in self.inp_nrns]
        self.out_nrns_inds = [self.nrn_names.index(nrn) for nrn in self.out_nrns]

        # compute the number of updates in a loop for running "animal.react" efficiently
        self.n_updates = self.blueprint.get_longest_path()
        self.grad_= jax.grad(self.get_performance_feedback, argnums=(0, 1))

    def finalize_action(self, raw_output):
        if self.action_type == 'Discrete':
            raw_output += self.action_noise * np.random.randn(*raw_output.shape)
            action = np.argmax(raw_output)
            return action
        elif self.action_type == 'Continuous':
            raw_output += self.action_noise * np.random.randn(*raw_output.shape)
            return np.clip(raw_output, self.action_bounds[0], self.action_bounds[1])
        elif self.action_type == 'MultiBinary':
            raw_output += self.action_noise * np.random.randn(*raw_output.shape)
            binarized_action = np.round(np.clip(raw_output, 0, 1), 0)
            return binarized_action
        elif self.action_type == 'Sigmoid':
            raw_output += self.action_noise * np.random.randn(*raw_output.shape)
            softmax_action = 1.0/(1 + np.exp(-raw_output))
            return softmax_action
        else:
            raise ValueError(f"action type {self.action_type} is not recognized!")


    def react(self, inputs):
        if len(inputs.shape) == 2:
            k = inputs.shape[1]
        else:
            k = 1
            inputs = inputs.reshape(-1, 1)
        N = self.W.shape[0]
        l = inputs.shape[0]

        nrn_vals = np.repeat(np.vstack([inputs, np.zeros((N-l, k))])[:, :, np.newaxis], self.n_updates+1, axis = 2)
        W_h = self.W[l:]
        b_h = self.b[l:]

        if self.n_updates == 0:
            return self.finalize_action(np.zeros((len(self.out_nrns_inds), k)))

        for i in range(self.n_updates-1):
            nrn_vals[l:, :, i + 1] = self.activation(W_h @ nrn_vals[:, :, i] + b_h[:, None])

        nrn_vals[l:, :, self.n_updates] = (W_h @ nrn_vals[:, :, self.n_updates-1])
        output = nrn_vals[self.out_nrns_inds, :, -1]
        return self.finalize_action(output)


    def mate(self, other_animal):
        fitness_parents = np.array([self.fitness, other_animal.fitness])
        neurons_parents = [self.blueprint.genome_dict["neurons"], other_animal.blueprint.genome_dict["neurons"]]
        synapses_parents = [self.blueprint.genome_dict["synapses"], other_animal.blueprint.genome_dict["synapses"]]

        neuron_names_parents = [set(neurons_parents[i].keys()) for i in range(2)]
        synapse_names_parents = [set(synapses_parents[i].keys()) for i in range(2)]

        genome_dict_child = {}
        genome_dict_child["neurons"] = deepcopy(recombine_genes(neurons_parents, neuron_names_parents, fitness_parents))
        genome_dict_child["synapses"] = deepcopy(recombine_genes(synapses_parents, synapse_names_parents, fitness_parents))

        n_inputs = self.blueprint.n_inputs
        n_outputs = self.blueprint.n_outputs
        blueprint_child = BluePrint(self.blueprint.innovation_handler,
                                    genome_dict_child,
                                    n_inputs, n_outputs,
                                    **self.blueprint_params)
        return blueprint_child


    def get_performance_feedback(self, W_h, b_h, inputs, targets, lmbd):
        N = W_h.shape[1]
        l = inputs.shape[0]
        k = inputs.shape[1]
        nrn_vals = jnp.repeat(jnp.expand_dims(jnp.vstack([inputs, jnp.zeros((N - l, k))]), axis=2),
                              repeats=self.n_updates + 1,
                              axis=2)
        if self.n_updates == 0:
            return jnp.sum((0.5 * jnp.ones_like(targets) - targets) ** 2) + lmbd * (
                        jnp.sum(W_h ** 2) + jnp.sum(b_h ** 2))

        for i in range(self.n_updates - 1):
            nrn_vals = nrn_vals.at[l:, :, i + 1].set(self.activation_jnp(W_h @ nrn_vals[:, :, i] + b_h[:, None]))
        nrn_vals = nrn_vals.at[l:, :, self.n_updates].set(W_h @ nrn_vals[:, :, self.n_updates - 1])

        # outputs = animal.finalize_action(nrn_vals[jnp.asarray(animal.out_nrns_inds), :, -1])
        raw_output = nrn_vals[jnp.asarray(self.out_nrns_inds), :, -1]
        outputs = 1.0 / (1 + jnp.exp(-raw_output))
        cost = jnp.sum((outputs - targets) ** 2) + lmbd * (jnp.sum(W_h ** 2) + jnp.sum(b_h ** 2))
        return cost

    # Define the update step function
    def learning_step(self, params, mask_W_h, inputs, targets, lmbd, lr):
        W_h, b_h = params
        gw, gb = self.grad_(W_h, b_h, inputs, targets, lmbd)
        new_W_h = W_h - (lr * gw) * mask_W_h
        new_b_h = b_h - lr * gb
        return new_W_h, new_b_h

    def live_and_learn(self, task, batch_size, lr, n_learning_episodes, lmbd):
        inputs, targets = task.get_batch(batch_size=batch_size)

        # before training
        W_bt = self.blueprint.get_connectivity_matrix()
        b_bt = self.blueprint.get_biases()

        l = len(inputs)
        # get the connectivity parts related to hidden and output neurons
        W_h = W_bt[l:]
        b_h = b_bt[l:]
        mask_W_h = (W_h != 0)

        def optimize(W_h, b_h, mask_W_h, inputs, targets, lmbd, lr):
            def scan_fn(params, _):
                return self.learning_step(params, mask_W_h, inputs, targets, lmbd, lr), 0

            final_params, _ = jax.lax.scan(scan_fn, (W_h, b_h), xs=None, length=n_learning_episodes)
            return final_params

        final_W_h, final_b_h = optimize(W_h, b_h, mask_W_h, inputs, targets, lmbd, lr)

        # assign new values to synapses
        W_at = np.array(jnp.concatenate([W_bt[:l, :], final_W_h], axis=0))
        b_at = np.array(jnp.concatenate([b_bt[:l], final_b_h], axis=0))
        return W_at, b_at



