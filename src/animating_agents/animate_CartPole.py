import json
from src.evolution.Blueprint import BluePrint
from src.evolution.Animal import Animal
from src.evolution.Nature_old import Nature
import gymnasium

env_name = 'CartPole-v1'
animals_param = {"neuron_type" : 'relu', "action_noise" : 0.03, "action_type" : "discrete"}
filename = f"../../data/evolved_models/{env_name}/CartPole-v1_top_animal_1113_07_02_2024.json"
file = open(filename, "rb")
data = json.load(file)
with file as json_file:
    for line in json_file:
        data = json.loads(line)

if "top_animal" in filename:
    genome_dict = data
else:
    genome_dict = data["genome_dict"]

env = gymnasium.make(env_name)

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n
blueprint = BluePrint(innovation_handler = None,
                      genome_dict=genome_dict,
                      n_inputs=n_inputs,
                      n_outputs=n_outputs)

animol = Animal(blueprint, **animals_param)
nature = Nature(innovation_handler=None,
                environment=env,
                logger=None)
nature.add_animal(animol)
res = nature.run_demonstration()
print(f"Total reward: {res}")


