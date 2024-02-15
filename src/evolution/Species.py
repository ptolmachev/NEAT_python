import numpy as np
from collections import deque

class Species():
    def __init__(self, representative_genome, species_id, c_w, c_d):
        """ A species requires at least one individual to come to existence """
        self.id = None
        self.age = 0
        self.subpopulation = []
        self.spawn_amount = 0
        self.no_improvement_age = 0
        self.__last_avg_fitness = 0
        self.representative_genome = representative_genome
        self.species_id =species_id
        self.c_w = c_w
        self.c_d = c_d
        self.age = 0
        L = 40
        self.top_fitness_list = deque(maxlen=L)
        self.std_fitness_list = deque(maxlen=L)
        self.mean_fitness_list = deque(maxlen=L)

    def add_animal(self, animal):
        animal.spec_id = self.id
        self.subpopulation.append(animal)
        return None

    def set_representative(self, animal):
        self.representative_genome = animal.blueprint.genome_dict

    def __mean(self):
        return np.mean(self.fitness_list)

    def __std(self):
        return np.std(self.fitness_list)

    def __len(self):
        return np.len(self.fitness_list)

    @property
    def fitness_list(self):
        return list([animal.fitness for animal in self.subpopulation])

    def get_gendist_to_representative(self, animal):
        genome1 = self.representative_genome
        genome2 = animal.blueprint.genome_dict
        return get_dist_btwn_genomes(genome1, genome2, self.c_w, self.c_d)


def get_dist_btwn_genomes(genome_dict_1, genome_dict_2, c_w, c_d):
    N1 = len(list(genome_dict_1["neurons"].keys()))
    N2 = len(list(genome_dict_2["neurons"].keys()))
    N = np.maximum(N1, N2)
    distance = 0
    disjoint_genes_count = 0

    # do counting for neurons
    innovs_1 = set(genome_dict_1["neurons"].keys())
    innovs_2 = set(genome_dict_2["neurons"].keys())
    innovs = list(innovs_1 | innovs_2)
    # go through innovations one by one and check if
    for innovation_id in innovs:
        if (innovation_id in innovs_1) and (innovation_id in innovs_2):
            b1 = genome_dict_1["neurons"][innovation_id]["bias"]
            b2 = genome_dict_2["neurons"][innovation_id]["bias"]
            b1 = 0 if b1 is None else b1
            b2 = 0 if b2 is None else b2
            distance += c_w * (np.abs(b1 - b2))
        else:
            disjoint_genes_count += 1

    # do counting for synapses
    innovs_1 = set(genome_dict_1["synapses"].keys())
    innovs_2 = set(genome_dict_2["synapses"].keys())
    innovs = list(innovs_1 | innovs_2)
    # go through innovations one by one and check if
    for innovation_id in innovs:
        if (innovation_id in innovs_1) and (innovation_id in innovs_2):
            w1 = genome_dict_1["synapses"][innovation_id]["weight"]
            w2 = genome_dict_2["synapses"][innovation_id]["weight"]
            distance += c_w * (np.abs(w1 - w2))
        else:
            disjoint_genes_count += 1
    distance += (c_d / N) * disjoint_genes_count
    return distance







