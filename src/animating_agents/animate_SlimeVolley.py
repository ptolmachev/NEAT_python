import json
from src.evolution.Blueprint import BluePrint
from src.evolution.Animal import Animal
from src.evolution.Nature import Nature
from src.slimevolleygym.slimevolley import SlimeVolleyEnv
import numpy as np
import time

env_name = 'SlimeVolley-v0'
max_timesteps = 3000
sleep = 0.01
animols_param = {"neuron_type" : 'relu', "action_noise" : 0.00, "action_type" : "MultiBinary", "action_bounds" : None}
filename = f"../../data/evolved_models/{env_name}/None_generation=584_score=0.7142857142857143_N=4.json"
file = open(filename, "rb")
data = json.load(file)
with file as json_file:
    for line in json_file:
        data = json.loads(line)
genome_dict = data["genome dict"]
genome_dict["synapses"] = {int(key): value for key, value in genome_dict["synapses"].items()}
genome_dict["neurons"] = {int(key): value for key, value in genome_dict["neurons"].items()}
for key in genome_dict["neurons"]:
    if genome_dict["neurons"][key]["type"] in ['i']:
        genome_dict["neurons"][key]["bias"] = 0.0
print(genome_dict["neurons"])
env = SlimeVolleyEnv()

n_inputs = env.observation_space.shape[0]
n_outputs = 3
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



