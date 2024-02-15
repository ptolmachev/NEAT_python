from collections import deque
from copy import deepcopy
from src.evolution.Blueprint import get_list_of_links, find_innovation_by_link


class InnovationHandler():
    '''This is a singleton class: at ensures that only one instance is created'''
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, maxlen=300):
        self.innovation_counter = 0
        self.maxlen = maxlen
        self.synapse_hash = deque(maxlen=int(self.maxlen))
        self.innov_ids = deque(maxlen=int(self.maxlen))

    def handle_innovation(self, genome_dict, innovation):
        synapse = (int(innovation["nrn_to"]), int(innovation["nrn_from"]))
        h = hash(synapse)

        # if this innovation recently appeared, find the corresponding innovation_id rather than assigning a new one
        if h in self.synapse_hash:
            innovation_id = self.innov_ids[self.synapse_hash.index(h)]
            genome_dict["synapses"][int(innovation_id)] = innovation
        else:
            # it might happen, that the genome_dict already have this innovation, but it got disabled.
            links = get_list_of_links(genome_dict, active_only=False)
            new_link = synapse
            for i in range(len(links)):
                if (links[i] == new_link):
                    old_innov_number = find_innovation_by_link(genome_dict=genome_dict, link=links[i])
                    genome_dict["synapses"][old_innov_number] = innovation
                    return genome_dict

            # only if everything above didn't realize, add new innovation
            genome_dict["synapses"][int(self.innovation_counter)] = innovation
            self.synapse_hash.append(h)
            self.innov_ids.append(int(self.innovation_counter))
            self.innovation_counter += 1
        return genome_dict


