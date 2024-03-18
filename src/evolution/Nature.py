import jax
from jax import lax, grad, hessian
import numpy as np
from jax import numpy as jnp
from functools import partial
from tqdm.auto import tqdm
import warnings
from itertools import chain
from src.evolution.Animal import Animal
from src.evolution.Blueprint import *
from src.evolution.Species import *
from copy import deepcopy
warnings.filterwarnings("ignore")
import ray


@ray.remote
class RemoteWorker():
    def __init__(self, nature, environment_builder, eval_repeats=11, max_timesteps=1000, metabolic_penalty=0.1):
        self.nature = nature
        self.environment = environment_builder()
        self.eval_repeats = eval_repeats
        self.max_timesteps = max_timesteps
        self.metabolic_penalty = metabolic_penalty

    def get_fitness(self, animal, seed, ref_animals):
        return self.nature.get_fitness(animal, seed, ref_animals)

    def get_vscore(self, animal, seed):
        return self.nature.get_vscore(animal, seed)

    def get_experienced_animal(self, animal, seed):
        return self.nature.get_experienced_animal(animal, seed)

class NatureBase():
    def __init__(self,
                 innovation_handler=None,
                 env_builder_fn=None,
                 env_name=None,
                 max_animals=32,
                 n_species_setpoint=15,
                 max_timesteps=1000,
                 blueprint_params=None,
                 animal_params=None,
                 rel_advantage_ind=3.0,
                 rel_advantage_spec=1.0,
                 mutation_probs=(0.03, 0.03, 0.2, 0.2, 0.5, 0.05),
                 perturb_biases=True,
                 syn_mutation_prob=0.8,
                 syn_mut_type_probs=(0.9, 0.05, 0.05),
                 weight_change_std=0.05,
                 c_d=1.0,
                 c_w=0.3,
                 delta=0.2,
                 gamma=1.1,
                 established_species_thr=5,
                 cull_ratio=0.2,
                 parthenogenesis_rate = 0.25,
                 interspecies_mating_chance=0.01,
                 metabolic_penalty = 0.05,
                 eval_repeats=11,
                 lifetime_learning=False,
                 n_learning_episodes = 0,
                 lr = 0.001,
                 validator=None,
                 logger=None,
                 save_logs = True,
                 log_every=10,
                 parallel=True,
                 n_workers=10,
                 self_play=False,
                 n_ref_animals=0):

        self.self_play = self_play
        if self.self_play:
            self.n_ref_animals = n_ref_animals
        self.max_timesteps = max_timesteps
        self.env_builder_fn = env_builder_fn
        self.environment = env_builder_fn()
        try:
            self.env_name = self.environment.spec.id
        except:
            self.env_name = env_name
        self.metabolic_penalty = metabolic_penalty
        self.mutation_probs = np.array(mutation_probs)
        self.syn_mutation_prob = np.array(syn_mutation_prob)
        self.syn_mut_type_probs = np.array(syn_mut_type_probs)
        self.weight_change_std = weight_change_std
        self.perturb_biases = perturb_biases
        self.innovation_handler = innovation_handler
        self.validator = validator
        self.logger = logger
        self.save_logs = save_logs
        self.log_every = log_every
        self.top_fitness_sequence = []
        self.mean_fitness_sequence = []
        self.std_fitness_sequence = []
        if not (self.validator is None):
            self.top_vscore_sequence = []
            self.mean_vscore_sequence = []
            self.std_vscore_sequence = []
        self.max_animals = max_animals
        self.n_species_setpoint = n_species_setpoint
        self.rel_advantage_ind = rel_advantage_ind
        self.rel_advantage_spec = rel_advantage_spec
        self.parthenogenesis_rate = parthenogenesis_rate
        self.interspecies_mating_chance = interspecies_mating_chance
        self.blueprint_params = blueprint_params
        self.animal_params = deepcopy(animal_params) # output type, neuron type, action bounds
        self.eval_repeats = eval_repeats
        self.lifetime_learning = lifetime_learning
        self.n_learning_episodes = n_learning_episodes
        self.lr = lr

        # params for evaluating distance between the genomes:
        self.c_w = c_w
        self.c_d = c_d
        self.delta = delta # speciation threshold
        self.gamma = gamma # parameter adjusting the speciation threshold if there are too many species
        self.established_species_thr = established_species_thr
        self.cull_ratio = cull_ratio
        self.current_generation = 0
        self.species_list = []
        self.species_counter = 0 #counts unique species over the whole evolution run
        self.parallel = parallel
        self.n_workers = n_workers


    def ensure_parallellism(self):
        @ray.remote
        def mate_parallel(animal1, animal2):
            return animal1.mate(animal2)

        self.mate_parallel = mate_parallel

        ray.shutdown()
        ray.init(num_cpus=self.n_workers)
        print(ray.available_resources())
        # send a function over
        self.remote_workers = [RemoteWorker.remote(nature=self,
                                                   environment_builder=self.env_builder_fn,
                                                   max_timesteps=self.max_timesteps,
                                                   metabolic_penalty=self.metabolic_penalty)
                               for _ in range(self.n_workers)]
        self.pool = ray.util.ActorPool(self.remote_workers)
        return None

    @property
    def fitness_list(self):
        return list(chain.from_iterable([species.fitness_list for species in self.species_list]))

    @property
    def vscore_list(self):
        return list(chain.from_iterable([species.vscore_list for species in self.species_list]))

    @property
    def animals(self):
        return list(chain.from_iterable([species.subpopulation for species in self.species_list]))

    def spawn_simplest_lifeforms(self, n):
        # get observation and actions spaces to put number of inputs and outputs
        n_inputs = self.environment.observation_space.shape[0]
        try:
            n_outputs = self.environment.action_space.n
        except:
            n_outputs = self.environment.action_space.shape[0]

        # create an empty neural graph
        genome_dict = {}
        genome_dict["neurons"] = {}
        genome_dict["synapses"] = {}
        for neuron in range(n_inputs):
            genome_dict["neurons"][self.innovation_handler.innovation_counter] = {"type": "i", "bias": 0.0}
            self.innovation_handler.innovation_counter += 1
        for neuron in range(n_outputs):
            genome_dict["neurons"][self.innovation_handler.innovation_counter] = {"type": "o", "bias": 0.0}
            self.innovation_handler.innovation_counter += 1

        b = BluePrint(innovation_handler=self.innovation_handler,
                      genome_dict=deepcopy(genome_dict),
                      n_inputs=n_inputs,
                      n_outputs=n_outputs,
                      **self.blueprint_params)
        # mutate these blueprints
        base_blueprints = [deepcopy(b) for i in range(n)]
        mutated_blueprints = [self.mutate(b) for b in base_blueprints]
        simplest_animals = [Animal(b, self.blueprint_params, **self.animal_params) for b in mutated_blueprints]
        return simplest_animals

    def get_fitness(self, animal, seed, ref_animals=None):
        raise NotImplementedError("This is a placeholder method of a base Nature class")

    def get_vscore(self, animal, seed):
        raise NotImplementedError("This is a placeholder method of a base Nature class")

    def get_experienced_animal(self, animal, seed):
        raise NotImplementedError("This is a placeholder method of a base Nature class")

    def mature_animals(self, animals, seed):
        if self.parallel:
            experienced_animals = list(self.pool.map(lambda W, a: W.get_experienced_animal.remote(animal=a, seed=seed), animals))
        else:
            experienced_animals = [self.get_experienced_animal(animal=animal, seed=seed) for animal in animals]
        return experienced_animals

    def eval_population(self):
        seed = self.current_generation
        refs = None
        if self.self_play:
            refs = np.random.choice(self.animals, self.n_ref_animals, replace=False)
        if self.parallel:
            fitness_vals = list(self.pool.map(lambda W, a: W.get_fitness.remote(animal=a, seed=seed, ref_animals=refs), self.animals))
        else:
            fitness_vals = [self.get_fitness(animal, seed, ref_animals=refs) for animal in self.animals]
        # assign each animal with the fitness value
        for i, animal in enumerate(self.animals):
            animal.fitness = fitness_vals[i]

        #the species has a property 'fitness_list' which dynamically pulls fitness from each animal
        for species in self.species_list:
            species.mean_fitness = np.nanmean(species.fitness_list)
            species.std_fitness = np.nanstd(species.fitness_list)

            # to store the history of values
            species.top_fitness_list.append(np.nanmax(species.fitness_list))
            species.mean_fitness_list.append(np.nanmean(species.fitness_list))
            species.std_fitness_list.append(np.nanstd(species.fitness_list))
        return None

    def validate_population(self):
        seed = self.current_generation + 10000
        if self.parallel:
            vscores = list(self.pool.map(lambda W, a: W.get_fitness.remote(animal=a, seed=seed, ref_animals=None), self.animals))
        else:
            vscores = [self.get_vscore(animal, seed) for animal in self.animals]

        # assign each animal with the value
        for i, animal in enumerate(self.animals):
            animal.vscore = vscores[i]

        for species in self.species_list:
            species.mean_vscore = np.nanmean(species.vscore_list)
            species.std_vscore = np.nanstd(species.vscore_list)

            # to store the history of values
            species.top_vscore_list.append(np.nanmax(species.vscore_list))
            species.mean_vscore_list.append(np.nanmean(species.vscore_list))
            species.std_vscore_list.append(np.nanstd(species.vscore_list))
        return None

    def mutate(self, blueprint):
        # possible mutations:
        # (0.03, 0.03, 0.2, 0.1, 0.8, 0.05) - default probabilities of mutation type occuring
        # (1) add neuron
        # (2) remove neuron
        # (3) add synapse
        # (4) remove synapse
        # (5) mutate weights (both synapses and biases)
        # (6) reset some bias to zero
        probs = self.mutation_probs
        perturb_weights = partial(blueprint.perturb_weights,
                                  mutation_prob=self.syn_mutation_prob,
                                  type_prob=self.syn_mut_type_probs,
                                  weight_change_std=self.weight_change_std,
                                  perturb_biases=self.perturb_biases)
        mutation_functions = [blueprint.add_neuron, blueprint.remove_neuron,
                              blueprint.add_synapse, blueprint.disable_synapse,
                              perturb_weights, blueprint.reset_bias]

        #random order of mutations
        mutation_ids = np.random.choice(np.arange(6), size=6, replace=False)
        [mutation_functions[int(id)]() for id in mutation_ids if np.random.rand() < probs[int(id)]]
        return blueprint

    def remove_old_generation(self):
        for species in self.species_list:
            for old_animal in species.subpopulation:
                del old_animal
            species.subpopulation.clear()
        return None

    def assign_species(self, animals):
        unassigned = animals
        assigned_to_existing_species = False
        while_cnt = 0
        while len(unassigned) != 0:
            if assigned_to_existing_species == False:
                # assigns animals to the existing species
                still_unassigned = []
                assigned = []
                SpDist = get_animal_to_species_DistMat(animals=unassigned, species_list=self.species_list)
                if not (SpDist is None):
                    for i, animal in enumerate(unassigned):
                        if np.min(SpDist[i, :]) <= self.delta:
                            species_ind = np.argmin(SpDist[i, :])
                            species = self.species_list[species_ind]
                            animal.species_id = species.species_id
                            species.subpopulation.append(animal)
                            assigned.append(animal)
                        else:
                            still_unassigned.append(animal)
                    unassigned = copy(still_unassigned)
                assigned_to_existing_species = True
            else:
                # create a new species and add it to the pool
                new_species = Species(representative_genome=unassigned[0].blueprint.genome_dict,
                                      species_id=self.species_counter,
                                      c_d = self.c_d, c_w=self.c_w)
                self.species_list.append(deepcopy(new_species))
                self.species_counter += 1
                assigned_to_existing_species = False

            while_cnt += 1
            if while_cnt > 1000:
                raise ValueError("Stuck in a loop in 'assign_species'!")

        self.species_list = [species for species in self.species_list if len(species.subpopulation) != 0]
        # if there are too many species, adjust the speciation threshold:
        lower_bound = int((3/4) * self.n_species_setpoint)
        upper_bound = int((4/3) * self.n_species_setpoint)

        if len(self.species_list) > upper_bound:
            self.delta *= self.gamma

        if len(self.species_list) < lower_bound:
            self.delta /= self.gamma
        return None

    def age_species(self):
        for species in self.species_list:
            species.age += 1
        return None

    def extinction_of_stagnant(self):
        survived_species = []
        for species in self.species_list:
            scores = species.top_fitness_list if (self.validator is None) else species.top_vscore_list
            l = scores.maxlen
            L = species.age
            if L >= l: #only if the species lived long enough
                if np.nanmean(list(scores)[:l//2]) < np.nanmean(list(scores)[l//2:]):
                    survived_species.append(species)
            else:
                survived_species.append(species)

        if len(survived_species) == 0:
            #all of the species are bad. Remove all species, allow only the top one to survive
            top_species_ind = np.nanargmax(np.array([np.max(species.fitness_list) for species in self.species_list]))
            top_species = self.species_list[top_species_ind]
            top_species.top_fitness_list.clear()
            top_species.mean_fitness_list.clear()
            top_species.std_fitness_list.clear()
            if not (self.validator is None):
                top_species.top_vscore_list.clear()
                top_species.mean_vscore_list.clear()
                top_species.std_vscore_list.clear()
            top_species.age = 0
            survived_species.append(top_species)

            # boost mutation rate:
            self.weight_change_std *= 1.05

            print("Increasing mutation rate!")

        for species in self.species_list:
            if not (species in survived_species):
                del species

        self.species_list = survived_species
        self.species_list = [species for species in self.species_list if len(species.subpopulation) != 0]
        #     # Alternatively, restart the whole process
        #     self.innovation_handler.innovation_counter = 0 # reset innovation handeler
        #     self.innovation_handler.synapse_hash = deque(maxlen=int(self.innovation_handler.maxlen))
        #     self.innovation_handler.innov_ids = deque(maxlen=int(self.innovation_handler.maxlen))
        #     simplest_animals = self.spawn_simplest_lifeforms(n=self.max_animals)
        #     self.assign_species(simplest_animals)
        #     self.eval_population()
        return None

    def assign_new_species_representative(self):
        for species in self.species_list:
            r = np.nanargmax(len(species.fitness_list)) # assign top animal to be its representatives
            species.representative_genome = deepcopy(species.subpopulation[r].blueprint.genome_dict)
        return None

    def cull_population(self):
        for species in self.species_list:
            subpop_fvals = species.fitness_list
            inds_animals_to_spare = np.where(subpop_fvals >= np.nanquantile(subpop_fvals, self.cull_ratio))[0]
            species.subpopulation = [species.subpopulation[ind] for ind in inds_animals_to_spare]
        return None

    def get_champions(self):
        # if species has more than "established_species_thr" individuals, it is considered to be experienced to draw champions.
        if len(self.species_list) == 1:
            return [deepcopy(self.species_list[0].subpopulation[np.nanargmax(self.species_list[0].fitness_list)].blueprint)]
        else:
            return [deepcopy(species.subpopulation[np.nanargmax(species.fitness_list)].blueprint)
                for species in self.species_list if len(species.subpopulation) > self.established_species_thr]

    def evolve_step(self):
        # remove unfit individuals from each species
        self.cull_population()
        # carry over one top individual from the every species without a change
        champions_blueprints = self.get_champions()
        n_champions = len(champions_blueprints)

        # fill up the rest of the spawned population with the children produced via crossover
        n_paris = self.max_animals - n_champions
        if n_paris <= 0:
            raise ValueError("The entire population has been spawned by existing champions."
                             " Try either to reduce number of n_species_setpoint or increase max_animals parameter!")
        pairs = self.get_mating_pairs(n_paris=n_paris)
        if self.parallel:
            childrens_blueprints = ray.get([self.mate_parallel.remote(self.animals[pair[0]], self.animals[pair[1]]) for pair in pairs])
        else:
            childrens_blueprints = [self.animals[pair[0]].mate(self.animals[pair[1]]) for pair in pairs]
        # mutations have to be centralized (because innovation handler can not be duplicated across multiple workers!)
        childrens_blueprints = [self.mutate(blueprint) for blueprint in childrens_blueprints]
        # add champions back, unaffected by any mutations
        childrens_blueprints.extend(champions_blueprints)
        spawned_animals = [Animal(b, self.blueprint_params, **self.animal_params) for b in childrens_blueprints]

        self.remove_old_generation() # empty the subpopulations
        if self.lifetime_learning:
            experienced_animals = self.mature_animals(spawned_animals, seed=np.random.randint(100000))
            self.assign_species(experienced_animals)
        else:
            self.assign_species(spawned_animals)
        self.eval_population()
        if not (self.validator is None):
            self.validate_population()

        self.age_species()
        self.extinction_of_stagnant()
        self.assign_new_species_representative()
        return None

    def run_evolution(self, n_generations):
        simplest_animals = self.spawn_simplest_lifeforms(n=self.max_animals)
        self.assign_species(simplest_animals)
        self.eval_population()
        self.best_score = -np.inf
        for i in range(n_generations):
            self.evolve_step()

            #save a bunch of stats
            self.top_fitness_sequence.append(np.nanmax(self.fitness_list))
            self.mean_fitness_sequence.append(np.nanmean(self.fitness_list))
            self.std_fitness_sequence.append(np.nanstd(self.fitness_list))
            if not (self.validator is None):
                self.top_vscore_sequence.append(np.nanmax(self.vscore_list))
                self.mean_vscore_sequence.append(np.nanmean(self.vscore_list))
                self.std_vscore_sequence.append(np.nanstd(self.vscore_list))

            if not (self.logger is None):
                self.log_information()
            self.current_generation += 1
        return None

    def log_information(self):
        tag = self.logger.tag if not (self.logger is None) else ''
        if not (self.validator is None):
            print(f"Generation {self.current_generation}; max vscore : {np.round(np.max(self.vscore_list), 3)}; max fitness : {np.round(np.max(self.fitness_list), 3)}")
        if self.save_logs and ((self.current_generation + 1) % self.log_every) == 0:
            log_dict = {}

            log_dict["top fitness score"] = float(self.top_fitness_sequence[-1])
            if not (self.validator is None):
                log_dict["top validation score"] = float(self.top_vscore_sequence[-1])

            log_dict["delta"] = float(self.delta)
            log_dict["N species"] = len(self.species_list)

            for species in self.species_list:
                if not (self.validator is None):
                    top_animal_within_species = species.subpopulation[np.nanargmax(species.vscore_list)]
                    log_dict[f"Species {species.species_id} top validation score"] = float(np.max(species.vscore_list))
                    log_dict[f"Species {species.species_id} mean validation score"] = float(np.mean(species.vscore_list))
                    log_dict[f"Species {species.species_id} std validation score"] = float(np.std(species.vscore_list))
                else:
                    top_animal_within_species = species.subpopulation[np.nanargmax(species.fitness_list)]
                top_animal_genome = top_animal_within_species.blueprint.genome_dict
                log_dict[f"Species {species.species_id} num animals"] = len(species.subpopulation)
                log_dict[f"Species {species.species_id} top fitness"] = float(np.max(species.fitness_list))
                log_dict[f"Species {species.species_id} mean fitness"] = float(species.mean_fitness)
                log_dict[f"Species {species.species_id} std fitness"] = float(species.std_fitness)

                log_dict[f"Species {species.species_id} N hidden neurons"] = len(get_neurons_by_type(top_animal_genome, type='h'))
                log_dict[f"Species {species.species_id} N synapses"] = len(list(top_animal_genome["synapses"].keys()))
                # log_dict[f"Species {species.species_id} top animal"] = top_animal_genome
            self.logger.save_log(log_dict, file_name=f"{self.env_name}_gen={self.current_generation}_{tag}.json")

        if (self.current_generation + 1) % self.logger.plot_every == 0:
            if not (self.validator is None):
                self.logger.plot_scores(top_scores=self.top_vscore_sequence,
                                        mean_scores=self.mean_vscore_sequence,
                                        std_scores=self.std_vscore_sequence,
                                        file_name=f"{self.env_name}_scores_{tag}.png")
            else:
                self.logger.plot_scores(top_scores=self.top_fitness_sequence,
                                        mean_scores=self.mean_fitness_sequence,
                                        std_scores=self.std_fitness_sequence,
                                        file_name=f"{self.env_name}_scores_{tag}.png")

        cur_score = self.top_fitness_sequence[-1] if (self.validator is None) else self.top_vscore_sequence[-1]
        if cur_score >= self.best_score:
            best_score_overall = cur_score
            top_animal = self.animals[np.nanargmax(self.vscore_list)]
            data_dict = deepcopy(top_animal.blueprint.genome_dict)
            if (self.validator is None):
                data_dict["score"] = best_score_overall
            else:
                data_dict["validation score"] = best_score_overall
            data_dict["N neurons"] = int(len(top_animal.blueprint.get_neurons_by_type("h")))
            data_dict["N synapses"] = int(len(list(top_animal.blueprint.genome_dict["synapses"].keys())))
            self.logger.fossilize(top_animal,
                                  generation=None,
                                  env_name=self.env_name,
                                  score=cur_score)
        return None

    def get_mating_pairs(self, n_paris):
        # based on the adjusted fitness of species, assign a probability of a child to be spawned by this species
        top_species_fitness = np.array([np.max(species.fitness_list) for species in self.species_list])
        top_species_rel_advantage = np.maximum(0, top_species_fitness - np.median(top_species_fitness))
        species_sizes = np.array([len(species.subpopulation) for species in self.species_list])
        adjusted_species_fitness = top_species_rel_advantage/species_sizes
        # probabilities of drawing a pair from a given species
        probs_species = get_probs(adjusted_species_fitness, slope=self.rel_advantage_spec)

        animal_species_ids = np.array([animal.species_id for animal in self.animals])
        animals_inds_per_species = [] # [[indices of animals in species 1], [indices of animals in species 2] ...]
        for species in self.species_list:
            animals_inds_per_species.append(np.where(animal_species_ids == species.species_id)[0].tolist())

        # sample mating pairs according to probabilities: randomly sampling the species first, then the animals within
        pairs = []
        for _ in range(n_paris):
            # sample the species first
            species_ind_1 = np.random.choice(np.arange(len(self.species_list)), p=probs_species)
            # inside the chosen species, select an animal based on its fitness
            fitness_vals_1 = self.species_list[species_ind_1].fitness_list
            probs_within_species_1 = get_probs(fitness_vals_1, slope=self.rel_advantage_ind)
            animal_ind_1 = np.random.choice(animals_inds_per_species[species_ind_1], p=probs_within_species_1)
            # scenario 1 - interspecies mating
            if (np.random.rand() < self.interspecies_mating_chance) and (len(self.species_list) > 1):
                other_species = list(range(len(self.species_list)))
                other_species = other_species[:species_ind_1] + other_species[species_ind_1 + 1:]

                #get the updated probabilities
                probs_species_rest = np.concatenate([probs_species[:species_ind_1],
                                                     probs_species[species_ind_1 + 1:]])
                probs_species_rest = probs_species_rest/np.sum(probs_species_rest)

                species_ind_2 = np.random.choice(other_species, p=probs_species_rest)

                # chose a second animal from the species 2:
                fitness_vals_2 = self.species_list[species_ind_2].fitness_list
                probs_within_species_2 = get_probs(fitness_vals_2, slope=self.rel_advantage_ind)

                animal_ind_2 = np.random.choice(animals_inds_per_species[species_ind_2], p=probs_within_species_2)
                pair = (animal_ind_1, animal_ind_2)
            else: #scenarios 2 and 3 - parthenogenesis and normal reproduction
                if len(fitness_vals_1) == 1 or np.random.rand() < self.parthenogenesis_rate:
                    pair = (animal_ind_1, animal_ind_1)
                else:
                    # get a new list of potential mates within the same species
                    potential_partners = copy(animals_inds_per_species[species_ind_1])
                    potential_partners = potential_partners[:animal_ind_1] + potential_partners[animal_ind_1 + 1:]
                    # get the updated probabilities
                    probs_within_species_1_rest = np.concatenate([probs_within_species_1[:animal_ind_1],
                                                                  probs_within_species_1[animal_ind_1 + 1:]])
                    probs_within_species_1_rest = probs_within_species_1_rest / np.sum(probs_within_species_1_rest)

                    animal_ind_2 = np.random.choice(potential_partners, p=probs_within_species_1_rest)
                    pair = (animal_ind_1, animal_ind_2)
            pairs.append(pair)
        return pairs


    def get_distance_matrix(self):
        D = np.zeros((len(self.animals), len(self.animals)))
        for i in range(len(self.animals)):
            for j in range(i + 1, len(self.animals)):
                D[i, j] = D[j, i] = get_dist_btwn_genomes(self.animals[i].blueprint.genome_dict,
                                                          self. animals[j].blueprint.genome_dict,
                                                          c_w=self.c_w, c_d=self.c_d)
        return D


class NatureOpenAIgym(NatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_environment(self, seed):
        try:
            obs = self.environment.reset(seed=seed)
        except:
            self.environment.seed(seed=seed)
            obs = self.environment.reset()
            self.environment.policy.reset()

        if type(obs) == tuple:  # for compatibility with other gym environments
            obs = obs[0]
        return obs

    def run_through_environment(self, animal, seed, other_animal=None):
        obs = self.prepare_environment(seed)
        if not (other_animal is None):
            obs_other = obs
        total_reward = 0
        done = False
        while_cnt = 0
        while (not done) and (while_cnt < self.max_timesteps):
            action = animal.react(inputs=obs)
            if not (other_animal is None):
                other_action = other_animal.react(obs_other)
                result = self.environment.step(action=action, otherAction=other_action)
            else:
                result = self.environment.step(action=action)
            match len(result):
                case 4: obs, reward, done, info = result
                case 5: obs, reward, done, _, info = result

            if not (other_animal is None):
                obs_other = info['otherObs']
            total_reward += reward
            while_cnt += 1
        return total_reward

    def get_fitness(self, animal, seed, ref_animals=None):
        if not (ref_animals is None):
            if len(ref_animals) == 0:
                raise ValueError("Need to set reference animals against which a primary animal is evaluated")

        rewards = []
        if not (ref_animals is None):
            for other_animal in ref_animals:
                for i in range(self.eval_repeats):
                    reward = self.run_through_environment(animal, seed=seed + i, other_animal=other_animal)
                    rewards.append(reward)
        else:
            for i in range(self.eval_repeats):
                reward = self.run_through_environment(animal, seed=seed + i, other_animal=None)
                rewards.append(reward)
        self.environment.close()

        if not (self.metabolic_penalty is None):
            W = animal.blueprint.get_connectivity_matrix()
            b = animal.blueprint.get_biases()
            penalty = self.metabolic_penalty * (np.sum(W ** 2) + np.sum(b) ** 2)
            return np.nanmean(rewards) - penalty
        return np.nanmean(rewards)

    def get_vscore(self, animal, seed):
        return self.validator.get_vscore(animal, seed)

class NatureCustomTask(NatureBase):
    def __init__(self, lifetime_learning=False, n_learning_episodes=500, lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.task = self.env_builder_fn()
        self.lifetime_learning = lifetime_learning
        self.lr = lr
        self.n_learning_episodes = n_learning_episodes

    def get_experienced_animal(self, animal, seed):
        # animal.blueprint.set_topological_order(animal.blueprint.genome_dict)
        return animal.live_and_learn(task=self.environment,
                                     batch_size=self.eval_repeats,
                                     lr=self.lr,
                                     lmbd=self.metabolic_penalty,
                                     n_learning_episodes=self.n_learning_episodes,
                                     seed=seed)

    def get_fitness(self, animal, seed, ref_animals=None):
        inputs, targets = self.task.get_batch(batch_size=self.eval_repeats, seed=seed)
        W = animal.blueprint.get_connectivity_matrix()
        b = animal.blueprint.get_biases()
        return -np.sum((animal.react(inputs) - targets)**2) - self.metabolic_penalty * (np.sum(W**2) + np.sum(b**2))


def get_animal_to_species_DistMat(animals, species_list):
    if len(species_list) == 0:
        return None
    SpDist = np.zeros((len(animals), len(species_list)))
    for i, animal in enumerate(animals):
        for j, species in enumerate(species_list):
            SpDist[i, j] = species.get_gendist_to_representative(animal)
    return SpDist

