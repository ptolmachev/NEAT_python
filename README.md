**NEAT: Neuroevolution of Augmenting Topologies**

In this algorithm multiple “animals” with the neural network defined by their genome evolve to solve a particular problem. 
For each animal, its “nervous system” (the neurons and their connectivity) is defined by the genome dictionary (see example below).

```yaml
{
  “neurons”: 
  {
    0: {“type” : “i”, “bias”: 0.0},
    1: {“type” : “o”, “bias”: 0.0},
     2: {“type” : “h”, “bias”: 0.0}
  },
  “Synapses”:
   {
    3 : {“nrn_to” : 2, “nrn_from”: 0, “actve”: True, “weight”: 1.0},
    4 : {“nrn_to” : 1, “nrn_from”: 2, “actve”: True, “weight”: 0.5}
  }
}

At first, the simplest network topologies are spawned, mutated, speciated and their performance is evaluated.
Further, the NEAT algorithm consists of iteratively cycling through phases in this order:

- Culling the population
-  Reproduction: spawning new animals
-   Mutation of the spawned genomes
-   Speciation of the animals
-   (Optional) Live and learn - given a fixed topology, modify the weights of the network via gradient descent
-   Evaluation of performance
-   Extinction of stagnant species

At (1) culling the population, for each species, the 25% of least fit animals are removed.

During the (2) reproduction phase, the old generation is replaced by the new one according the following rules:
The champions (the most fit individuals from the species) from each species with at least 5 individuals are carried forward to the new generation unaffected (this will ensure the upward trend in improving the performance of the population).
The rest of the spots available (N - N_champions) for new animals (the population is kept constant for simplicity, with N animals) according to the following procedure:
For each new animal, a first parent needed to be chosen first. A species from which the parent is coming from is sampled randomly (see details later). The first parent is then randomly sampled from the sampled species (see details later). 
With probability “parthenogenesis_rate” = 0.25, the chosen animal carries its genome to the next generation unaffected, mimicking sexless reproduction. 
With probability 1 - ”parthenogenesis_rate”, another animal has to be selected for mating. Normally, another animal-parent is chosen from the same species, however, with probability “interspecies_mating_rate” a different species is chosen (in the same manner as before), and another animal-parent is chosen from the new species. 
Given the two animal-parents, the crossover of the two genomes is done according to the following rules.
If a given gene is present in both parents, then the inherited gene is chosen with 50/50 probability (even if the genes are both code for the same neurons or synapse, the specific weight or bias may be different).
For the disjoint genes, the gene to inherit is chosen from the most fit individual. If the two parents are equally fit, the choice is random with 50/50 chance.

To choose either species or individual for further reproduction, the fitness of a given individual/species has to be translated into probability of choosing it for further reproduction.
This is done by the following procedure.

Suppose we have a list of p fitness values. First, we calculate the median fitness over these p values (F). Then we use the following formula for assigning the probabilities of choosing:
A = [(f_1 - F),(f_2 - F), ..., (f_p - F)]+,
where [...]+ denotes taking only positive values, while setting the negative values to 0.
B = [e^{-a_1}, e^{-a_2}, ..., e^{-a_p}],
where the λ parameter controls the relative fitness advantage, and is set to 1.0 for choosing the species, and 3.0 if choosing an individual within the species.
Finally, the probabilities of choosing species/individual for reproduction are calculated as Pi = BiiBi.
Fitness values of the species are calculated as the maximal fitness value within the species divided by the size of the population. Such a penalty prevents the domination of a single species, which, in turn, makes the search for the network topology wider and more efficient.

During the (3) mutation phase, several de novo mutations may take place:

Adding a neuron:
With probability 0.03 by default, a random synapse is chosen and disabled.
In between the two neurons “nrn_from” and “nrn_to” a “new_nrn” is inserted, and the two synapses are added into the genome: 
{“nrn_to” : nrn_to, “nrn_from” : new_nrn, weight: old_weight, “active”: True}
{“nrn_to” : “new_nrn”, “nrn_from” : “nrn_from”, weight: 1.0, “active”: True}
Note that the first synapse in the chain is always set with the weight 1.0, whereas the second synapses is set with the “old_weight” of the disabled synapse. This is done to preserve the performance as much as possible, while given room for new useful mutations.
If no synapse is yet present in the genome dictionary, the mutation does nothing.
Removing an orphaned neuron:
With probability p = 0.03, if the network has a neuron which doesn’t connect to anything else, it is removed from the genome. If no such neurons are present, the mutation does nothing.
Adding a synapse:
With probability 0.3, two neurons are chosen, in such a way that the synapse between them would not produce a cycle (handled via topologically sorting the neurons). A new synapse between these two neurons is added with a new weight, randomly sampled from a normal distribution with mu = 0, sigma=0.4. If no new synapse can be added, mutation does nothing.
Removing a synapse:
With probability 0.3, an existing synapse is chosen and disabled (setting “active” to False). The synapses with lesser absolute weights are preferred (simulating atrophy).
If there are no active synapses, the mutation does nothing. 
Mutate weights of synapses: 
Taking 80% of all the synapses and perturbing them as follows.
For each synapse marked for perturbation, with 90% probability a given synapse is perturbed with normal random variable with mu=0 and sigma = 0.1, with 5% probability the existing weight is doubled, and with 5% probability it is halved)

For the (4) evaluation phase, all the animals are evaluated on task n_repeat times with different seeds (the n_repeats seeds are kept the same for all the animals, so that they go through the same trials in parallel), and assigned with the fitness value equal to the average score they attain over the n_repeat times.
[The evaluation of multiple animals is parallelized for efficiency with ray]

(5) live an learn:
For a given topology and the given initial weight, an animal is trained to perform the task with backpropagation. For now, this step works only for the tasks for which targets are available (in a supervised manner, so that the error can be computed). 
In principle, it is possible to make this step work with reinforcement learning as well.

During the (6) speciation phase, the animals are assigned to a species based on their proximity to the representative genome of the species. 
The proximity of two given genomes is measured as the sum of two terms:
The number of the disjoint genes (Nd) divided by the number of genes in the longest genome (N) taken with the coefficient c_d = 1.0: c_d * Nd/N
The average weights and biases difference for the genes which a present in both genomes taken with coefficient c_w: c_w * sum (|wi - wi’|)

Each newly spawned and mutated genome is assigned to species if the distance between the given genome and a representative genome of the species is below a certain threshold (delta = 3.0). If, however, the genome in question is far away from all existing species, a new species is created with the new genome set as a representative genome.
Further, for each species, a new representative genome is chosen from the genomes of animals currently assigned to this species.

Finally, in the (7) extinction of the stagnant phase, if a given species has failed to improve its top fitness for specified time, it is removed from the species with all the animals in it, simulating an extinction.

The full list of hyperparameters for the NEAT algorithm is summarized in the config file for a given task (the precise parameters may vary from task to task).

The relevant code is implemented in the ‘evolution’ subdirectory.

Performance:

LunarLander-v2 OpenAI gym task

![LunarLander-v2 performance](https://github.com/ptolmachev/NEAT_python/blob/main/img/LunarLander-v2/LunarLander-v2_scores_1445_04_03_2024.png)

The genome contains only one hidden neuron:
(During all the training I didn’t allow biases)

```yaml
"genome dict": {
    "neurons": {
    "0": {"type": "i","bias": 0.0},
    "1": { "type": "i", "bias": 0.0},
    "2": {"type": "i", "bias": 0.0},
    "3": {"type": "i", "bias": 0.0},
    "4": { "type": "i","bias": 0.0},
    "5": {"type": "i","bias": 0.0},
    "6": {"type": "i","bias": 0.0},
    "7": {"type": "i","bias": 0.0},
    "8": {"type": "o","bias": 0.0},
    "9": {"type": "o","bias": 0.0},
    "10": {"type": "o","bias": 0.0},
    "11": {"type": "o","bias": 0.0},
    "299": {"type": "h","bias": 0.0}
},
    "synapses": {
    "33": {"nrn_to": 10,"nrn_from": 3,"weight": -0.46037378719559485, "active": true},
    "39": {"nrn_to": 10,"nrn_from": 1,"weight": -0.39984998921543174, "active": true},
    "330": {"nrn_to": 299,"nrn_from": 3,"weight": 0.2956668480402732, "active": true},
    "12": {"nrn_to": 9,"nrn_from": 6,"weight": -0.6179245653204352, "active": false},
    "300": {"nrn_to": 299,"nrn_from": 6,"weight": 0.4749252003551596,"active": false},
    "301": {"nrn_to": 9,"nrn_from": 299,"weight": -1.6570164653792536, "active": true},
    "45": {"nrn_to": 8,"nrn_from": 7,"weight": 0.12432028625503151, "active": true},
    "16": {"nrn_to": 9,"nrn_from": 2,"weight": 0.25961601949407387, "active": true},
    "23": {"nrn_to": 9,"nrn_from": 5,"weight": -1.0340246558599957, "active": true},
    "27": {"nrn_to": 10,"nrn_from": 6,"weight": -0.4388951455678786, "active": false},
    "30": {"nrn_to": 11,"nrn_from": 4,"weight": 0.9505962211272357, "active": true},
    "31": {"nrn_to": 11,"nrn_from": 7,"weight": -0.9274080082782241, "active": false},
    "440": {"nrn_to": 11,"nrn_from": 0,"weight": -0.6026475237481741, "active": true}
}


In addition, the NEAT is applied to solve several supervised tasks (XOR, classifying Moons, Circles and Spirals) with just a few hidden neurons.
Below are some plots of the decision boundaries:


![Spirals](https://github.com/ptolmachev/NEAT_python/blob/main/img/Spirals/Spirals%20result.png)
![Moons](https://github.com/ptolmachev/NEAT_python/blob/main/img/Moons/Moons%20result.png)
![Circles](https://github.com/ptolmachev/NEAT_python/blob/main/img/Circles/circles%20result.png)
![XOR](https://github.com/ptolmachev/NEAT_python/blob/main/img/XOR/XOR%20result.png)

Further directions:

Keeping hard-coded speciation might not be necessary. Instead, one can introduce a soft-speciation, by utilizing an “affinity function”, which accepts the distance between the two genomes and returns the likelihood of the two animals mating. The larger the distance, the less likely the two animals mate with one another.
In addition, one can introduce that the two closely related animals (having the same genes) will be less likely to mate as well, nudging the evolution towards greater diversification within the species, making the search more efficient. 
Fun fact (!): about 20% of marriages are between cousins. 80% of all marriages in history have been between second cousins or closer.

Additionally, I become really interested in how the real biological chromosomes align: perhaps, one can draw further inspiration on how to create biologically plausible crossovers, and handle the speciation by borrowing ideas from biological mitosis.

The hyperparameters for the evolution are set manually and are by no means optimal.
It is quite straightforward to use such packages as optuna to optimize the hyperparameters (which implement Bayesian optimization strategy). However, running hyperparameters optimization is quite time consuming, and it is reserved more for an end product.

Current backpropagation is a bit hacky, because sometimes the gradients returns nans. For now, if the optimization doesn’t converge, I keep reducing the learning rate and try one more time for 10 times (maximal learning rate reduction is thus by a factor of 1024). However, sometimes, the optimization still doesn’t converge. In that case, I revert to the unoptimized weights and biases.
Spending more time writing an optimizer which does all the checks will certainly make life much easier. 
Since jax can also compute hessians, it might be a good idea to do a second order optimization. However, computing the hessian and the inverse of it (to use in Newton-Raphson method) might be even more time consuming than doing, say, 2000 steps of gradient descent. 
There is always a way to optimize the code. For instance, during the performance evaluation, each animal has to compute an output given the input multiple times, making it the most called function in the evolution loop, calling it approximately n_timesteps x n_eval_repeats x n_animals x n_generations times. Optimizing this function would directly translate into speed up of the algorithm. Currently, while computing the output of the neural network, I identified the longest path it takes from input to reach the output, and then just iteratively multiplied the connectivity matrix with the neural activity followed by the application of the activation function. In practice it works fine (faster than looping through each neuron in the topological order), however, looping through seems more computationally efficient (at least in theory). Making the looping through neurons practically might be a good idea to pursue to speed up the computations.

There definitely should be a way to use GPU cells to massively parallelize computations, but I haven’t looked through this.
