import graphlib
from functools import partial
from graphlib import TopologicalSorter
from src.evolution.Blueprint import get_neural_graph
import numpy as np
import hydra
from src.evolution.Blueprint import *
from src.evolution.Animal import *
import json
import time

# filename = f"../../data/evolved_models/SlimeVolley-v0/None_generation=812_score=-4.2_N=16.json"
# file = open(filename, "rb")
# data = json.load(file)
# with file as json_file:
#     for line in json_file:
#         data = json.loads(line)
# genome_dict = data["genome dict"]
# genome_dict["synapses"] = {int(key): value for key, value in genome_dict["synapses"].items()}
# genome_dict["neurons"] = {int(key): value for key, value in genome_dict["neurons"].items()}
from src.evolution.InnovationHandler import InnovationHandler


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

ni = 8
no = 4
nh = 6
tic = time.time()
for _ in range(100):
    genome_dict = get_random_genome(ni, no, nh)
    IH = InnovationHandler()
    IH.innovation_counter = 100000
    blueprint = BluePrint(genome_dict=genome_dict, innovation_handler=IH, n_inputs=ni, n_outputs=no)

    #mutate:
    perturb_weights = partial(blueprint.perturb_weights,
                              mutation_prob=0.8,
                              type_prob=[0.9, 0.05, 0.05],
                              weight_change_std=0.05,
                              perturb_biases=True)

    mutation_functions = [blueprint.add_neuron, blueprint.remove_neuron,
                          blueprint.add_synapse, blueprint.disable_synapse,
                          perturb_weights, blueprint.reset_bias]
    probs = np.array([1,1,1,1,1,1])
    probs= probs/np.sum(probs)
    for i in range(100):
        [mutation_functions[i]() for i in range(len(probs)) if np.random.rand() < probs[i]]

    animal = Animal(blueprint, action_type="Continuous", action_noise=0, action_bounds=(-1, 1))
    for i in range(500):
        inputs = np.random.randn(ni)
        animal.react(inputs)

toc = time.time()

print(toc - tic)


# graph = get_neural_graph(blueprint.genome_dict, active_only=True)
# topsorter = TopologicalSorter(graph)
# inp_nrns = get_neurons_by_type(genome_dict, 'i')
# out_nrns = get_neurons_by_type(genome_dict, 'o')
# sorted_nrns = list(topsorter.static_order())
# sorted_nrns = [nrn for nrn in sorted_nrns if not ((nrn in inp_nrns) or (nrn in out_nrns))]
# print(f"sorted hidden neurons: {sorted_nrns}")
# print(graph)

