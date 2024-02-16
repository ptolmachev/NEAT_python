import json
import time
import gymnasium
from src.evolution.Blueprint import BluePrint
from src.evolution.Animal import Animal
from src.evolution.Logger import Logger
import numpy as np

env_name = 'Pendulum-v1'
animols_param = {"neuron_type" : 'relu', "action_noise" : 0.00, "action_type" : "Continuous"}
filename = f"../../data/evolved_models/{env_name}/Pendulum-v1_generation=655_score=-154.14406767246913_N=6.json"
file = open(filename, "rb")
data = json.load(file)
with file as json_file:
    for line in json_file:
        data = json.loads(line)

max_timesteps = 500
sleep = 0.001
genome_dict = data["genome dict"]
env = gymnasium.make(env_name, render_mode='human')

def convert_keys_to_int(dictionary):
    return {int(key): value for key, value in dictionary.items()}

genome_dict["synapses"] = convert_keys_to_int(genome_dict["synapses"])
genome_dict["neurons"] = convert_keys_to_int(genome_dict["neurons"])

n_inputs = env.observation_space.shape[0]
try:
    n_outputs = env.action_space.n
except:
    n_outputs = env.action_space.shape[0]
blueprint = BluePrint(innovation_handler = None,
                      genome_dict=genome_dict,
                      n_inputs=n_inputs,
                      n_outputs=n_outputs)

animal = Animal(blueprint, **animols_param)

seed = np.random.randint(100000)
try:
    obs = env.reset(seed=seed)
except:
    obs = env.reset()
    env.seed(seed=seed)

env.render()
if type(obs) == tuple:  # for compatibility with other gym environments
    obs = obs[0]
total_reward = 0

for i in range(max_timesteps):
    env.render()
    time.sleep(sleep)
    action = animal.react(inputs=obs)
    result = env.step(action=action)
    match len(result):
        case 3: obs, reward, done = result
        case 4: obs, reward, done, info = result
        case 5:  obs, reward, done, _, info = result
    total_reward += reward
    if done:
        break
env.close()

print(f"Total reward: {total_reward}")


