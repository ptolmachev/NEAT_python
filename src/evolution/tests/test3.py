import numpy as np
from src.evolution.Blueprint import BluePrint
from src.evolution.InnovationHandler import InnovationHandler
import time

def get_random_genome(ni, no, nh):
    # make connections from input neurons to output neurons
    inp_neurons = list(range(ni))
    out_neurons = list(range(inp_neurons[-1]+1, inp_neurons[-1] + 1 + no))
    hidden_neurons = list(range(out_neurons[-1]+1, out_neurons[-1] + 1 + nh))
    neurons = {}
    for inp_nrn in inp_neurons:
        neurons[inp_nrn] = {"type":"i", "bias": 0}
    for out_nrn in out_neurons:
        neurons[out_nrn] = {"type":"o", "bias": 0}
    for h_nrn in hidden_neurons:
        neurons[h_nrn] = {"type":"h", "bias": np.random.randn()}
    synapses = {}
    odds = np.array([ni * nh, ni * no, nh * no, nh * (nh - 1) / 2.0])
    probs = odds/np.sum(odds)
    for i in range(100):
        # sample the scenario:
        scenario = int(np.random.choice(np.arange(4), p=probs))
        if scenario == 0:
            # input -> hidden
            nrn_from = int(np.random.choice(inp_neurons))
            nrn_to = int(np.random.choice(hidden_neurons))
        elif scenario == 1:
            # inp -> output
            nrn_from = int(np.random.choice(inp_neurons))
            nrn_to = int(np.random.choice(out_neurons))
        elif scenario == 2:
            # hidden -> output
            nrn_from = int(np.random.choice(hidden_neurons))
            nrn_to = int(np.random.choice(out_neurons))
        elif scenario == 3:
            # hid -> hid
            nrn_from = int(np.random.choice(hidden_neurons))
            nrn_to = int(np.random.choice(hidden_neurons))
        synapses[i+100] = {"nrn_to" : nrn_to, "nrn_from" : nrn_from, "weight" : np.random.randn(), "active" : True}
    # set up a blueprint
    genome_dict = {}
    genome_dict["neurons"] = neurons
    genome_dict["synapses"] = synapses
    return genome_dict


def activation(x):
    return np.maximum(0, x)

def compute_outputs_1(inputs, W, b, inp_nrns_inds, out_nrns_inds):
    nrn_vals = np.zeros(W.shape[0])
    nrn_vals[inp_nrns_inds] = inputs
    l = len(inputs)
    W = W[l:, :]
    b = b[l:]
    nrn_vals[:len(inputs)] = inputs
    for i in range(10000):
        nrn_vals_prev = np.copy(nrn_vals)
        nrn_vals[l:] = activation(W @ nrn_vals_prev + b)
        if np.array_equal(nrn_vals_prev, nrn_vals):
            output = nrn_vals[list(out_nrns_inds)]
            return output


