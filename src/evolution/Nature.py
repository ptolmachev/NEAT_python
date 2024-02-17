import numpy as np
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
class RemoteEnvironment():
    def __init__(self, environment_builder, eval_repeats=11, max_timesteps=1000, metabolic_penalty=0.1):
        self.environment = environment_builder()
        self.eval_repeats = eval_repeats
        self.max_timesteps = max_timesteps
        self.metabolic_penalty = metabolic_penalty

    def prepare_environment(self, seed):
        try:
            obs = self.environment.reset(seed=seed)
        except:
            self.environment.seed(seed=seed)
            obs = self.environment.reset()
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

    def get_fitness(self, animal, seed, reference_animals=None):
        if not (reference_animals is None):
            if len(reference_animals) == 0:
                raise ValueError("You need to assign a list of other animals against whom you evaluate your promary animal")

        rewards = []
        if not (reference_animals is None):
            for other_animal in reference_animals:
                for i in range(self.eval_repeats):
                    reward = self.run_through_environment(animal, seed=seed + i, other_animal=other_animal)
                    rewards.append(reward)
        else:
            for i in range(self.eval_repeats):
                reward = self.run_through_environment(animal, seed=seed + i)
                rewards.append(reward)
        self.environment.close()

        if not (self.metabolic_penalty is None):
            W = animal.blueprint.get_connectivity_matrix()
            b = animal.blueprint.get_biases()
            penalty = self.metabolic_penalty * (np.sum(W ** 2) + np.sum(b) ** 2)
            return np.nanmean(rewards) - penalty
        return np.nanmean(rewards)

class Nature():
    def __init__(self,
                 innovation_handler,
                 environment_builder_fn,
                 max_animals=32,
                 max_species=6,
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
                 logger=None,
                 save_logs = True,
                 log_every=10,
                 num_workers=10,
                 parallel=True,
                 self_play=False,
                 num_reference_animals=0,
                 validator=None):
        self.self_play = self_play
        self.num_reference_animals = num_reference_animals
        self.parallel = parallel
        self.max_timesteps = max_timesteps
        self.environment_builder_fn = environment_builder_fn
        self.environment = environment_builder_fn()
        try:
            self.env_name = self.environment.spec.id
        except:
            self.env_name = None
        self.num_workers = num_workers
        self.metabolic_penalty = metabolic_penalty
        self.mutation_probs = np.array(mutation_probs)
        self.syn_mutation_prob = np.array(syn_mutation_prob)
        self.syn_mut_type_probs = np.array(syn_mut_type_probs)
        self.weight_change_std = weight_change_std
        self.perturb_biases = perturb_biases
        self.innovation_handler = innovation_handler
        self.logger = logger
        self.save_logs = save_logs
        self.log_every = log_every
        self.validator = validator
        self.max_animals = max_animals
        self.max_species = max_species
        self.rel_advantage_ind = rel_advantage_ind
        self.rel_advantage_spec = rel_advantage_spec
        self.parthenogenesis_rate = parthenogenesis_rate
        self.interspecies_mating_chance = interspecies_mating_chance
        self.blueprint_params = blueprint_params
        if "action_bounds" in list(animal_params.keys()):
            animal_params["action_bounds"] = animal_params["action_bounds"]
        self.animal_params = deepcopy(animal_params) # output type, neuron type, action bounds
        self.eval_repeats = eval_repeats
        # params for evaluating distance between the genomes:
        self.c_w = c_w
        self.c_d = c_d
        self.delta = delta #speciation threshold
        self.gamma = gamma # parameter adjusting the speciation threshold if there are too many species
        self.established_species_thr = established_species_thr
        self.cull_ratio = cull_ratio
        self.current_generation = 0
        self.species_list = []
        self.species_counter = 0 #counts unique species over the whole evolution run

        if self.parallel:
            ray.shutdown()
            ray.init(num_cpus=self.num_workers)
            print(ray.available_resources())
            # send a function over
            self.remote_envs = [RemoteEnvironment.remote(environment_builder=self.environment_builder_fn,
                                                         max_timesteps=max_timesteps,
                                                         metabolic_penalty=self.metabolic_penalty)
                                for _ in range(self.num_workers)]
            self.pool = ray.util.ActorPool(self.remote_envs)

            @ray.remote
            def mate_parallel(animal1, animal2):
                return animal1.mate(animal2)
            self.mate_parallel = mate_parallel

    @property
    def fitness_list(self):
        return list(chain.from_iterable([species.fitness_list for species in self.species_list]))

    @property
    def animals(self):
        return list(chain.from_iterable([species.subpopulation for species in self.species_list]))

    def spawn_simplest_lifeforms(self):
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
        base_blueprints = [deepcopy(b) for i in range(self.max_animals)]
        mutated_blueprints = [self.mutate(b) for b in base_blueprints]
        simplest_animals = [Animal(b, **self.animal_params) for b in mutated_blueprints]
        return simplest_animals

    def prepare_environment(self, seed):
        try:
            obs = self.environment.reset(seed=seed)
        except:
            self.environment.seed(seed=seed)
            obs = self.environment.reset()
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

    def get_fitness(self, animal, seed, reference_animals=None):
        if not (reference_animals is None):
            if not reference_animals:
                raise ValueError("You need to assign a list of reference animals against which you evaluate the primary animal")

        rewards = []
        if not (reference_animals is None):
            for other_animal in reference_animals:
                for i in range(self.eval_repeats):
                    reward = self.run_through_environment(animal, seed=seed + i, other_animal=other_animal)
                    rewards.append(reward)
        else:
            for i in range(self.eval_repeats):
                reward = self.run_through_environment(animal, seed=seed+i)
                rewards.append(reward)
        self.environment.close()

        if not (self.metabolic_penalty is None):
            W = animal.blueprint.get_connectivity_matrix()
            b = animal.blueprint.get_biases()
            penalty = self.metabolic_penalty * (np.sum(W ** 2) + np.sum(b) ** 2)
            return np.nanmean(rewards) - penalty
        return np.nanmean(rewards)

    def eval_population(self):
        # seed = self.current_generation
        seed = np.random.randint(100000)
        if self.self_play:
            reference_animals = np.random.choice(self.animals, self.num_reference_animals, replace=False)
        else:
            reference_animals = None

        if self.parallel:
            fitness_vals = list(self.pool.map(lambda a, animal: a.get_fitness.remote(animal, seed, reference_animals=reference_animals), self.animals))
        else:
            fitness_vals = [self.get_fitness(animal, seed, reference_animals=reference_animals) for animal in self.animals]

        # assign each animal with the fitness value
        for i, animal in enumerate(self.animals):
            animal.fitness = fitness_vals[i]

        self.mean_fitness = np.mean(np.array(fitness_vals))
        self.std_fitness = np.std(np.array(fitness_vals))

        #the species has a property 'fitness_list' which dynamically pulls fitness from each animal
        for species in self.species_list:
            species.mean_fitness = np.mean(species.fitness_list)
            species.std_fitness = np.std(species.fitness_list)
            # to store the history of values
            species.top_fitness_list.append(np.max(species.fitness_list))
            species.mean_fitness_list.append(np.mean(species.fitness_list))
            species.std_fitness_list.append(np.std(species.fitness_list))
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

        for i in range(len(probs)):
            if np.random.rand() < probs[i]:
                mutation_functions[i]()
        return blueprint

    def remove_old_generation(self):
        for species in self.species_list:
            for old_animal in species.subpopulation:
                del old_animal
            species.subpopulation.clear()
        return None

    def speciate(self, animals):
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
                raise ValueError("Stuck in a loop in speciate!")

        # delete the species which have zero progeny in it.
        species_to_remove = [species for species in self.species_list if len(species.subpopulation) == 0]
        self.species_list = list(set(self.species_list) - set(species_to_remove))
        # if there are too many species, adjust the speciation threshold:
        if len(self.species_list) >= self.max_species:
            self.delta *= self.gamma
        return None

    def age_species(self):
        for species in self.species_list:
            species.age += 1
        return None

    def extinction_of_stagnant(self):
        survived_species = []
        for species in self.species_list:
            L = species.age
            if L == species.top_fitness_list.maxlen:
                top_fitness_list = list(species.top_fitness_list)
                if np.mean(top_fitness_list[L//2:]) > np.mean(top_fitness_list[:L//2]):
                    survived_species.append(species)
                # top_fitness_list = list(species.top_fitness_list)
                # mean_fitness_list = list(species.mean_fitness_list)
                # std_fitness_list = list(species.std_fitness_list)
                # # if there is an improvement in top fitness
                # clause_1 = np.mean(top_fitness_list[L//2:]) > np.mean(top_fitness_list[:L//2])
                # # if there is an improvement in mean fitness
                # clause_2 = np.mean(mean_fitness_list[L//2:]) > np.mean(mean_fitness_list[:L//2])
                # # if there the std of fitness inside the populations decreased
                # clause_3 = np.mean(std_fitness_list[L//2:]) < np.mean(std_fitness_list[:L//2])
                # # if at least either of this happens, the species is not stagnant
                # if clause_1 or clause_2 or clause_3:
                #     survived_species.append(species)
            else:
                survived_species.append(species)
        if len(survived_species) == 0:
            #jeeez...all of the species are bad. perhaps allow only the top one to survive
            top_species_ind = np.argmax(np.array([np.max(species.fitness_list) for species in self.species_list]))
            top_species = self.species_list[top_species_ind]
            top_species.top_fitness_list.clear()
            top_species.mean_fitness_list.clear()
            top_species.std_fitness_list.clear()
            survived_species.append(top_species)
            # boost mutation rate:
            self.weight_change_std *= 2.0
            print("Increasing mutation rate!")

        for species in self.species_list:
            if not (species in survived_species):
                del species
        self.species_list = survived_species
        return None

    def assign_new_species_representative(self):
        for species in self.species_list:
            r = np.argmax(len(species.fitness_list)) # assign top animal to be its representatives
            species.representative_genome = deepcopy(species.subpopulation[r].blueprint.genome_dict)
        return None

    def cull_population(self):
        for species in self.species_list:
            subpop_fvals = species.fitness_list
            inds_animals_to_spare = np.where(subpop_fvals >= np.quantile(subpop_fvals, self.cull_ratio))[0]
            species.subpopulation = [species.subpopulation[ind] for ind in inds_animals_to_spare]
        return None

    def get_champions(self):
        # if species has more than "established_species_thr" individuals, it is considered to be mature to draw champions.
        return [deepcopy(species.subpopulation[np.argmax(species.fitness_list)].blueprint)
                for species in self.species_list if len(species.subpopulation) > self.established_species_thr]

    def evolve_step(self):
        # remove unfit individuals from each species
        self.cull_population()
        # carry over one top individual from the every species without a change
        champions_blueprints = self.get_champions()
        n_champions = len(champions_blueprints)

        # fill up the rest of the spawned population with the children produced via crossover
        num_pairs = self.max_animals - n_champions
        if num_pairs <= 0:
            raise ValueError("The entire population has been spawned by existing champions."
                             " Try either to reduce number of max_species or increase max_animals parameter!")
        pairs = self.get_mating_pairs(num_pairs=num_pairs)
        if self.parallel:
            childrens_blueprints = ray.get([self.mate_parallel.remote(self.animals[pair[0]], self.animals[pair[1]]) for pair in pairs])
        else:
            childrens_blueprints = [self.animals[pair[0]].mate(self.animals[pair[1]]) for pair in pairs]
        # mutations have to be centralized (because innovation handler can not be duplicated across multiple workers!)
        childrens_blueprints = [self.mutate(blueprint) for blueprint in childrens_blueprints]
        # add champions back, unaffected by any mutations
        childrens_blueprints.extend(champions_blueprints)
        spawned_animals = [Animal(b, **self.animal_params) for b in childrens_blueprints]

        self.remove_old_generation() # empty the subpopulations
        self.speciate(spawned_animals)
        self.eval_population()
        self.age_species()
        self.extinction_of_stagnant()
        self.assign_new_species_representative()
        return np.max(self.fitness_list)

    def evolve_loop(self, n_generations):
        best_score_overall = -np.inf
        top_score_sequence = []
        val_scores_sequence = [] if not (self.validator is None) else None
        mean_score_sequence = []
        std_score_sequence = []

        simplest_animals = self.spawn_simplest_lifeforms()
        self.speciate(simplest_animals)
        self.eval_population()

        for i in tqdm(range(n_generations)):
            self.evolve_step()
            fitness_vals = np.array(self.fitness_list)

            if not (self.validator is None):
                cur_val_score = self.validator.get_validation_score(animal=self.animals[np.argmax(fitness_vals)],
                                                                    )
                val_scores_sequence.append(cur_val_score)

            cur_top_score = np.max(fitness_vals)
            top_score_sequence.append(cur_top_score)
            mean_score_sequence.append(np.mean(fitness_vals))
            std_score_sequence.append(np.std(fitness_vals))

            if not (self.logger is None):
                tag = self.logger.tag if not (self.logger is None) else ''
                if self.save_logs and ((self.current_generation + 1) % self.log_every) == 0:
                    log_dict = {}
                    log_dict[f"Num species"] = len(self.species_list)
                    for species in self.species_list:
                        log_dict[f"Species {species.species_id} num animals"] = len(species.subpopulation)
                        log_dict[f"Species {species.species_id} top fitness"] = np.max(species.fitness_list)
                        log_dict[f"Species {species.species_id} mean fitness"] = species.mean_fitness
                        log_dict[f"Species {species.species_id} std fitness"] = species.std_fitness
                        top_animal_genome = species.subpopulation[np.argmax(species.fitness_list)].blueprint.genome_dict
                        log_dict[f"Species {species.species_id} top animal N hidden neurons"] = \
                            len(get_neurons_by_type(top_animal_genome, type='h'))
                        log_dict[f"Species {species.species_id} top animal N synapses"] = \
                            len(list(top_animal_genome["synapses"].keys()))
                        log_dict[f"Species {species.species_id} top animal"] = top_animal_genome

                    self.logger.save_log(log_dict, file_name=f"{self.env_name}_gen={self.current_generation}_{tag}.json")

                if (self.current_generation + 1) % self.logger.plot_every == 0:
                    self.logger.plot_scores(top_scores=top_score_sequence,
                                            val_scores=val_scores_sequence,
                                            mean_scores=mean_score_sequence,
                                            std_scores=std_score_sequence,
                                            file_name=f"{self.env_name}_scores_{tag}.png")

                cur_score = cur_top_score if (self.validator is None) else cur_val_score
                if cur_score >= best_score_overall:
                    best_score_overall = cur_score
                    top_animal = self.animals[np.argmax(fitness_vals)]
                    data_dict = deepcopy(top_animal.blueprint.genome_dict)
                    if (self.validator is None):
                        data_dict["score"] = best_score_overall
                    else:
                        data_dict["val score"] = best_score_overall
                    data_dict["N_neurons"] = int(len(top_animal.blueprint.get_neurons_by_type("h")))
                    data_dict["N_synapses"] = int(len(list(top_animal.blueprint.genome_dict["synapses"].keys())))
                    self.logger.fossilize(top_animal, self.current_generation, self.env_name, score=best_score_overall)
            self.current_generation += 1
        # save the top animal at the end
        # self.logger.fossilize(top_animal, self.current_generation, self.env_name, score=best_score_overall)
        return None

    # def get_mating_pairs(self, num_pairs):
    #     # based on the adjusted fitness of species, assign a probability of a child to be spawned by this species
    #     top_species_fitness = np.array([np.max(species.fitness_list) for species in self.species_list])
    #     top_species_rel_advantage = np.maximum(0, top_species_fitness - np.median(top_species_fitness))
    #     species_sizes = np.array([len(species.subpopulation) for species in self.species_list])
    #     adjusted_species_fitness = top_species_rel_advantage/species_sizes
    #     # probabilities of drawing a pair from a given species
    #     probs_species = get_probs(adjusted_species_fitness, slope=self.rel_advantage_spec)
    #
    #     animal_species_ids = np.array([animal.species_id for animal in self.animals])
    #     animals_inds_per_species = [] # [[indices of animals in species 1], [indices of animals in species 2] ...]
    #     for species in self.species_list:
    #         animals_inds_per_species.append(np.where(animal_species_ids == species.species_id)[0].tolist())
    #
    #
    #     # sample mating pairs according to probabilities: randomly sampling the species first, then the animals within
    #     pairs = []
    #     for _ in range(num_pairs):
    #         # sample the species first
    #         species_ind = np.random.choice(np.arange(len(self.species_list)), p=probs_species)
    #         # inside the chosen species, select an animal based on its fitness
    #         fitness_vals = standardize(self.species_list[species_ind].fitness_list)
    #         if len(self.species_list[species_ind].subpopulation) == 1:
    #             #reproduce via parthenogenesis, cause there is only one animal in this species
    #             animal_ind = animals_inds_per_species[species_ind][0]
    #             pair = (animal_ind, animal_ind)
    #         else:
    #             probs_within_species = get_probs(fitness_vals, slope=self.rel_advantage_ind)
    #             if np.random.rand() > self.parthenogenesis_rate:
    #                 pair = tuple(np.random.choice(animals_inds_per_species[species_ind],
    #                                         size=2, p=probs_within_species,
    #                                         replace=False))
    #             else:
    #                 # reproduce via parthenogenesis
    #                 hermaphrodite_ind = np.random.choice(animals_inds_per_species[species_ind], p=probs_within_species)
    #                 pair = (hermaphrodite_ind, hermaphrodite_ind)
    #         pairs.append(pair)
    #     return pairs

    def get_mating_pairs(self, num_pairs):
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
        for _ in range(num_pairs):
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

def get_animal_to_species_DistMat(animals, species_list):
    if len(species_list) == 0:
        return None
    SpDist = np.zeros((len(animals), len(species_list)))
    for i, animal in enumerate(animals):
        for j, species in enumerate(species_list):
            SpDist[i, j] = species.get_gendist_to_representative(animal)
    return SpDist

