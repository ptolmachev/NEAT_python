import itertools
from copy import deepcopy, copy
import numpy as np
from src.evolution.utils import standardize, get_probs
from collections import OrderedDict
from collections import defaultdict
from graphlib import TopologicalSorter

class BluePrint():
    def __init__(self, innovation_handler, genome_dict, n_inputs, n_outputs,
                 weight_init_std=0.4,
                 orph_node_thr=0.1,
                 max_weight_val=3.0,
                 max_neurons=32):
        '''The genome dict has to contain the following information
        1) a list of nodes
        2) a list of synapses with innovation_id, active or disabled, nrn_to id, nrn_from id, weight'''

        # (1) check if there are invalid synapses
        links = get_list_of_links(genome_dict)
        input_nrns = get_neurons_by_type(genome_dict, type='i')
        output_nrns = get_neurons_by_type(genome_dict, type='o')
        self.n_inputs = n_inputs
        self.inp_neurons = input_nrns
        self.n_outputs = n_outputs
        self.out_neurons = output_nrns

        for nrn_to, nrn_from in links:
            if nrn_to in input_nrns:
                raise ValueError(
                    "There is a connection coming TO an input neuron in the initialization of a blueprint!")
            if nrn_from in output_nrns:
                raise ValueError(
                    "There is a connection coming FROM an output neuron in the initialization of a blueprint!")
        # (2) check whether there are duplicates:
        if len(links) != len(set(links)):
            genome_dict = remove_duplicate_synapse(genome_dict)
        # (3) check if there are orphaned neurons and synapses
        genome_dict = remove_orphaned_neurons(genome_dict)
        genome_dict = remove_orphaned_synapses(genome_dict)
        success = self.set_topological_order(genome_dict)
        # (4) remove cycles if present
        if success == False:
            genome_dict = remove_cycles(genome_dict)
            success = self.set_topological_order(genome_dict)
        if success == False:
            print("Something is definitely wrong")
        self.genome_dict = genome_dict

        self.innovation_handler = innovation_handler
        self.orph_node_thr = orph_node_thr
        self.max_neruons = max_neurons
        self.max_weight_val = max_weight_val
        self.weight_init_std = weight_init_std

    def get_neurons_by_type(self, type):
        return get_neurons_by_type(self.genome_dict, type)

    def get_connectivity_matrix(self):
        return get_connectivity_matrix(genome_dict=self.genome_dict)

    def get_adjacency_matrix(self):
        return get_connectivity_matrix(genome_dict=self.genome_dict, adjacency_only=True)

    def get_biases(self):
        return get_biases(self.genome_dict)

    def get_list_of_links(self, active_only=True):
        return get_list_of_links(self.genome_dict, active_only=active_only)

    def set_topological_order(self, genome_dict):
        top_sorter = TopologicalSorter(get_neural_graph(genome_dict))
        res = top_sorter._find_cycle()
        cycle = None if res is None else res[:-1]
        if not (cycle is None):
            return False
        else:
            top_order_full = list(top_sorter.static_order())
            self.topological_order = [nrn for nrn in top_order_full if not ((nrn in self.inp_neurons) or (nrn in self.out_neurons))]
            # self.topological_order = []
            # for nrn in top_order_full:
            #     if not ((nrn in self.inp_neurons) or (nrn in self.out_neurons)):
            #         self.topological_order.append(nrn)
        return True

    def add_neuron(self):
        # do not add any more neurons if there are too many of them
        if len(self.get_neurons_by_type(type="h")) >= self.max_neruons:
            return False

        # get all the active genes
        active_genes = {gene: info for gene, info in self.genome_dict["synapses"].items() if info["active"]}
        # if there are no active genes (that means there are no synapses)
        if len(list(active_genes.keys())) == 0:
            return False

        # add extra neuron to hidden neuron list in the genome_dict
        new_nrn = self.innovation_handler.innovation_counter
        self.genome_dict["neurons"][new_nrn] = {"type": "h", "bias" : 0.0}
        self.innovation_handler.innovation_counter += 1

        # sample from active genes randomly
        rnd_ind = np.random.randint(len(active_genes))
        innov_number_to_disable = int(list(active_genes.keys())[rnd_ind])

        # disable it and add two new innovations, with new ids
        self.genome_dict["synapses"][innov_number_to_disable]["active"] = False
        synapse_info = self.genome_dict["synapses"][innov_number_to_disable]
        nrn_to, nrn_from, old_weight = synapse_info["nrn_to"], synapse_info["nrn_from"], synapse_info["weight"]

        # add two new innovations
        innovation_1 = {"active": True,
                       "nrn_to" : int(new_nrn),
                       "nrn_from" : int(nrn_from),
                       "weight" : 1.0} # new weights are always 1.0

        innovation_2 = {"active": True,
                       "nrn_to" : int(nrn_to),
                       "nrn_from" : int(new_nrn),
                       "weight" : old_weight}

        self.genome_dict = self.innovation_handler.handle_innovation(genome_dict=self.genome_dict,
                                                                     innovation=innovation_1)
        self.genome_dict = self.innovation_handler.handle_innovation(genome_dict=self.genome_dict,
                                                                     innovation=innovation_2)

        # update_topological order manually, insert new node right after the nrn_from
        if int(nrn_from) in self.inp_neurons:
            self.topological_order.insert(0, int(new_nrn)) #if the nrn_from was an input nrn then put new_nrn first
        elif int(nrn_to) in self.out_neurons:
            self.topological_order.append(int(new_nrn))
        else:
            self.topological_order.insert(self.topological_order.index(int(nrn_from)) + 1, int(new_nrn))
        return True

    def remove_neuron(self):
        # line up neurons which are orphaned
        types = [neuron_info["type"] for neuron_info in  self.genome_dict["neurons"].values()]
        neuron_names = list(map(int, self.genome_dict["neurons"].keys()))
        N = len(types)
        W = self.get_connectivity_matrix()
        biases = self.get_biases()
        net_input_connections = np.sum(np.abs(np.hstack([W, biases.reshape(-1, 1)])), axis=1)
        mask = np.array([True if ((net_input_connections[i] < self.orph_node_thr) and (types[i] == 'h')) else False for  i in range(N)])
        orphaned_neurons = [neuron_names[ind] for ind in (np.where(mask == True)[0])]
        if not orphaned_neurons:
            return False
        nrn_to_remove = int(np.random.choice(orphaned_neurons, p=get_probs(-net_input_connections[mask], slope=10)))
        neuron_names.remove(nrn_to_remove)
        self.topological_order.remove(nrn_to_remove)

        # add all the neurons back apart from the removed one
        genome_dict_new = {}
        genome_dict_new["neurons"] = {name: self.genome_dict["neurons"][name] for name in neuron_names}

        # remove all the synapses associated with the removed neuron
        genome_dict_new["synapses"] = {
            synapse_id: synapse_info
            for synapse_id, synapse_info in self.genome_dict["synapses"].items()
            if synapse_info["nrn_to"] != nrn_to_remove and synapse_info["nrn_from"] != nrn_to_remove
        }
        self.genome_dict = genome_dict_new
        return True

    def add_synapse(self):
        hid_neurons = self.get_neurons_by_type(type='h')
        list_of_links = self.get_list_of_links()
        # it could be four different scenarios: (inp-> hid), (inp-> out), (hid->hid), (hid->out)
        ni = len(self.inp_neurons)
        nh = len(hid_neurons)
        no = len(self.out_neurons)
        odds = np.array([ni * nh, ni * no, nh * no, nh * (nh - 1) / 2.0])
        probs = odds/np.sum(odds)
        for i in range(100):
            #sample the scenario:
            scenario = int(np.random.choice(np.arange(4), p=probs))
            if scenario == 0:
                # input -> hidden
                nrn_from = int(np.random.choice(self.inp_neurons))
                nrn_to = int(np.random.choice(hid_neurons))
            elif scenario == 1:
                #inp -> output
                nrn_from = int(np.random.choice(self.inp_neurons))
                nrn_to = int(np.random.choice(self.out_neurons))
            elif scenario == 2:
                # hidden -> output
                nrn_from = int(np.random.choice(self.topological_order))
                nrn_to = int(np.random.choice(self.out_neurons))
            elif scenario == 3:
                # hid -> hid
                sampled_nrns_inds = np.random.choice(np.arange(len(self.topological_order)), size=2, replace=False)
                if sampled_nrns_inds[0] > sampled_nrns_inds[1]:
                    sampled_nrns_inds = sampled_nrns_inds[::-1]
                nrn_to = int(self.topological_order[sampled_nrns_inds[1]])
                nrn_from = int(self.topological_order[sampled_nrns_inds[0]])

            # Check if the synapse already exists
            if ((nrn_to, nrn_from) in list_of_links):
                continue
            #otherwise put new synapse in
            innovation = {"nrn_to": nrn_to, "nrn_from": nrn_from,
                          "weight": self.weight_init_std * np.random.randn(),
                          "active": True}
            self.genome_dict = self.innovation_handler.handle_innovation(genome_dict=self.genome_dict,
                                                                         innovation=innovation)
            return True
        return False #if it fails after 100 attempts

    def perturb_weights(self, mutation_prob=0.8,
                        type_prob=(0.9, 0.05, 0.05),
                        weight_change_std=0.05,
                        perturb_biases=True):

        gene_types = ["synapses"]
        if perturb_biases == True:
            gene_types.append("neurons")
        for gene_type in gene_types:
            if gene_type == "neurons": # allow biases only on hidden units to change
                innovations = [innovation for innovation, info in self.genome_dict[gene_type].items() if (info["type"] == 'h')]
            else:
                innovations = self.genome_dict[gene_type].keys()
            if len(innovations) == 0:
                continue

            key = "weight" if gene_type == "synapses" else "bias"

            mutate_mask = np.random.rand(len(innovations)) <= mutation_prob  # 80% are mutated
            for innovation, mutate in zip(innovations, mutate_mask):
                if mutate:
                    r = np.random.choice(np.arange(3), p=type_prob)
                    old_weight = self.genome_dict[gene_type][innovation][key]
                    if r == 0: new_weight = old_weight + weight_change_std * np.random.randn()
                    elif r == 1: new_weight = old_weight / 2.0
                    else: new_weight = old_weight * 2.0
                    self.genome_dict[gene_type][innovation][key] = np.clip(new_weight, -self.max_weight_val, self.max_weight_val)
        return True

    def disable_synapse(self):
        innovations = list(self.genome_dict["synapses"].keys())
        if not innovations:
            return False

        weights = np.array([self.genome_dict["synapses"][innovation]["weight"] for innovation in innovations])
        if len(weights) >= 6:
            # if the synapses is relatively useless it is more likely to be deleted
            probs = get_probs(-np.abs(weights), slope=10)
            sampled_innovation = int(np.random.choice(innovations, p=probs))
            self.genome_dict["synapses"][sampled_innovation]["active"] = False
        return True

    def reset_bias(self):
        hidden_neurons = self.get_neurons_by_type(type='h')
        if len(hidden_neurons) == 0:
            return False
        self.genome_dict["neurons"][int(np.random.choice(hidden_neurons))]["bias"] = 0.0
        return True

    def get_longest_path(self):
        l = self.n_inputs
        A = self.get_adjacency_matrix()[l:]
        nrn_vals = np.zeros(A.shape[1])
        nrn_vals[:l] = np.ones(l)
        nrn_vals_prev = np.zeros_like(nrn_vals)
        for i in range(10000):
            nrn_vals_prev = np.copy(nrn_vals)
            nrn_vals[l:] = A @ nrn_vals_prev
            if np.array_equal(nrn_vals_prev, nrn_vals):
                return i + 1
        raise ValueError("There is a loop in the connectivity!")

#########################EXTERNAL FUNCTIONS########################

def recombine_genes(genes_by_parents, genes_names_by_parent, fitness):
    probs = [0.9, 0.1] if (fitness[0] > fitness[1]) else ([0.5, 0.5] if (fitness[0] == fitness[1]) else [0.1, 0.9])
    unique_gene_names = set(genes_names_by_parent[0]) | set(genes_names_by_parent[1])
    child_gene = {}
    for name in unique_gene_names:
        if name in genes_names_by_parent[0] and name in genes_names_by_parent[1]:
            parent_index = np.random.choice([0, 1], p=probs)
            child_gene[name] = genes_by_parents[parent_index][name]
        elif name in genes_names_by_parent[0] and fitness[0] > fitness[1]:
            child_gene[name] = genes_by_parents[0][name]
        elif name in genes_names_by_parent[1] and fitness[1] > fitness[0]:
            child_gene[name] = genes_by_parents[1][name]
    return child_gene

def remove_duplicate_synapse(genome_dict):
    synapses = genome_dict["synapses"]
    connections = [(info["nrn_to"], info["nrn_from"]) for synapse, info in synapses.items()]
    if len(set(connections)) == len(connections):
        return genome_dict

    innovations = list(synapses.keys())
    clashing_innovations = []
    for i in range(len(connections)):
        for j in range(i + 1, len(connections)):
            if connections[i] == connections[j]:
                clashing_innovations.append((innovations[i], innovations[j]))

    innovations_to_remove = [np.random.choice(clash) for clash in clashing_innovations]
    innovations_to_have = list(set(innovations) - set(innovations_to_remove))
    new_genome_dict = {}
    new_genome_dict["synapses"] = {innovation: genome_dict["synapses"][innovation] for innovation in innovations_to_have}
    new_genome_dict["neurons"] = genome_dict["neurons"]
    return new_genome_dict


def remove_orphaned_neurons(genome_dict):
    participating_neurons = set(itertools.chain(*get_list_of_links(genome_dict)))
    neuron_names = genome_dict["neurons"].keys()
    neurons_to_delete = [name for name in neuron_names if
                         genome_dict["neurons"][name]["type"] not in {'i', 'o'} and name not in participating_neurons]

    if not neurons_to_delete:
        return genome_dict

    genome_dict_new = {
        "synapses": genome_dict["synapses"],
        "neurons": {name: genome_dict["neurons"][name] for name in neuron_names if name not in neurons_to_delete}
    }
    return genome_dict_new

def remove_orphaned_synapses(genome_dict):
    neuron_names = set(genome_dict["neurons"].keys())
    links = get_list_of_links(genome_dict, active_only=False)
    innovations_to_remove = [find_innovation_by_link(link, genome_dict) for link in links if not all(n in neuron_names for n in link)]
    innovations_to_have = set(genome_dict["synapses"].keys()) - set(innovations_to_remove)
    genome_dict_new = {
        "neurons": genome_dict["neurons"],
        "synapses": {innov: genome_dict["synapses"][innov] for innov in innovations_to_have}
    }
    return genome_dict_new

def remove_cycles(genome_dict):
    neural_graph = get_neural_graph(genome_dict, active_only=True)
    top_sorter = TopologicalSorter(neural_graph)
    res = top_sorter._find_cycle()
    cycle = None if res is None else res[:-1]
    while not (cycle is None):
        tuple_chain = [(cycle[(i+1) % len(cycle)], cycle[i]) for i in range(len(cycle))]
        rnd_ind = np.random.randint(len(tuple_chain))
        link_to_silence = tuple_chain[rnd_ind]
        innov_to_silence = find_innovation_by_link(link_to_silence, genome_dict)
        genome_dict["synapses"][innov_to_silence]["active"] = False
        nrn_to, nrn_from = int(link_to_silence[0]), int(link_to_silence[1])
        neural_graph[nrn_to].remove(nrn_from)
        # neural_graph = get_neural_graph(genome_dict, active_only=True)
        top_sorter = TopologicalSorter(neural_graph)
        res = top_sorter._find_cycle()
        cycle = None if res is None else res[:-1]
    return genome_dict

def find_innovation_by_link(link, genome_dict):
    for innovation, synapse_info in genome_dict["synapses"].items():
        if (synapse_info["nrn_to"], synapse_info["nrn_from"]) == link:
            return innovation
    return None

def get_neural_graph(genome_dict, active_only=True):
    graph = defaultdict(list)
    for neuron in genome_dict['neurons'].keys():
        graph[int(neuron)] = []
    for synapse_info in genome_dict['synapses'].values():
        if (active_only and synapse_info["active"]) or (active_only == False):
            nrn_from = str(synapse_info['nrn_from'])
            nrn_to = str(synapse_info['nrn_to'])
            graph[int(nrn_to)].append(int(nrn_from)) #uppend incoming connections!
    return graph

def get_connectivity_matrix(genome_dict, adjacency_only=False):
    synapses = genome_dict["synapses"]
    nrn_names = list(map(int, genome_dict["neurons"].keys()))
    n_nrns = len(nrn_names)
    W = np.zeros((n_nrns, n_nrns))
    for synapse_info in synapses.values():
        if synapse_info["active"]:
            nrn_to = int(synapse_info["nrn_to"])
            nrn_from = int(synapse_info["nrn_from"])
            weight = synapse_info["weight"]
            if nrn_to in nrn_names and nrn_from in nrn_names:
                idx_to = nrn_names.index(nrn_to)
                idx_from = nrn_names.index(nrn_from)
                W[idx_to, idx_from] = weight if not adjacency_only else 1.0
    return W

def get_biases(genome_dict):
    return np.array([neuron_info['bias'] for neuron_info in genome_dict["neurons"].values()])

def get_neurons_by_type(genome_dict, type):
    return [int(name) for name, info in genome_dict["neurons"].items() if info["type"] == type]

def get_list_of_links(genome_dict, active_only=True):
    if active_only:
        return [(s["nrn_to"], s["nrn_from"]) for s in genome_dict["synapses"].values() if s["active"]]
    else:
        return [(s["nrn_to"], s["nrn_from"]) for s in genome_dict["synapses"].values()]

