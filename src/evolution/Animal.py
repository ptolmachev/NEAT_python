from copy import deepcopy
import numpy as np
from src.evolution.Blueprint import *

class Animal():
    def __init__(self, blueprint, neuron_type='relu', action_noise=0.03, action_type='Discrete', action_bounds=(-1, 1)):
        self.blueprint = deepcopy(blueprint)
        self.W = self.blueprint.get_connectivity_matrix()
        self.b = self.blueprint.get_biases()
        self.action_noise = action_noise
        self.n_inputs = self.blueprint.n_inputs
        self.n_outputs = self.blueprint.n_outputs
        self.neuron_type = neuron_type
        self.neural_activity = None
        match neuron_type:
            case "sigmoid": self.activation = lambda x: 1.0 / (np.exp(x) + 1.0)
            case "relu": self.activation = lambda x: np.maximum(0, x)
            case "tanh": self.activation = lambda x: np.tanh(x)
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
        self.n_updates = self.blueprint.get_longest_path() - 1

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
        else:
            raise ValueError(f"action type {self.action_type} is not recognized!")
    #
    # def react(self, inputs):
    #     nrn_vals = np.zeros(self.W.shape[0])
    #     nrn_vals[:len(inputs)] = inputs
    #     nrn_vals_prev = np.zeros_like(nrn_vals)
    #     l = len(inputs)
    #     W = self.W[l:]
    #     b = self.b[l:]
    #
    #     for i in range(10000):
    #         nrn_vals_prev = np.copy(nrn_vals)
    #         nrn_vals[l:] = self.activation(W @ nrn_vals_prev + b)
    #         if np.array_equal(nrn_vals_prev, nrn_vals):
    #             output = nrn_vals[list(self.out_nrns_inds)]
    #             return self.finalize_action(output)
    #     raise ValueError("There likely is a loop in connectivity!")

    def react(self, inputs):
        nrn_vals = np.zeros(self.W.shape[0])
        l = len(inputs)
        nrn_vals[:l] = inputs
        W = self.W[l:]
        b = self.b[l:]
        for i in range(self.n_updates):
            nrn_vals[l:] = self.activation(W @ nrn_vals + b)
        output = nrn_vals[list(self.out_nrns_inds)]
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
        blueprint_child = BluePrint(self.blueprint.innovation_handler, genome_dict_child, n_inputs, n_outputs)
        return blueprint_child


