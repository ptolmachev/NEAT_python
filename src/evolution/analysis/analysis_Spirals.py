import json
from src.evolution.Blueprint import BluePrint
from src.evolution.Animal import Animal
from matplotlib import pyplot as plt
import numpy as np
from src.evolution.Tasks.Tasks import TaskSpirals
np.set_printoptions(suppress=True)
env_name = 'Spirals'
filename = f"../../../data/evolved_models/{env_name}/Spirals_score=-23.082726892065093_N=8.json"
# filename = f"../../../data/evolved_models/{env_name}/Spirals_score=-26.295079904756644_N=7.json"

file = open(filename, "rb")
data = json.load(file)
with file as json_file:
    for line in json_file:
        data = json.loads(line)
genome_dict = data["genome dict"]
genome_dict["neurons"] = {int(id): info for id, info in genome_dict["neurons"].items()}
blueprint = BluePrint(n_inputs=3, n_outputs=1, genome_dict=genome_dict, innovation_handler=None)
animal = Animal(blueprint,
                action_noise=0,
                action_type="Sigmoid",
                action_bounds=None,
                blueprint_params=None,
                neuron_type="relu")

batch_size = 1000
task = TaskSpirals()

inputs_scatter, targets_scatter = task.get_batch(batch_size=batch_size, seed = np.random.randint(10000))
outputs_scatter = (animal.react(inputs_scatter).flatten())

print(f"MSE naive {np.sum((outputs_scatter - targets_scatter)**2)}")


#generate a grid
n = 100
inputs_grid = np.zeros((3, n**2))
targets_grid = np.zeros(n**2)
inputs_grid[2, :] = 1.0
x = np.linspace(-1.25, 1.25, n)
cnt = 0
X, Y = np.meshgrid(x, x)
inputs_grid[0, :] = X.flatten()
inputs_grid[1, :] = Y.flatten()

output_grid = animal.react(inputs_grid)
output_grid = output_grid.reshape(n, n)
fig = plt.plot(figsize = (5, 5))

plt.imshow(output_grid[::-1, :], extent=(np.min(x), np.max(x), np.min(x), np.max(x)),
           cmap='bwr', interpolation='bilinear',
           vmin=0, vmax=1, alpha=0.4)
plt.colorbar()  # Add color bar for reference

inds_pm = np.where(outputs_scatter > 0.5)[0]
inds_nm = np.where(outputs_scatter < 0.5)[0]
alpha_pm = outputs_scatter[inds_pm]
alpha_nm = 1 - outputs_scatter[inds_nm]
if len(inds_pm) != 0:
    plt.scatter(inputs_scatter[0, inds_pm], inputs_scatter[1, inds_pm], color = 'r', edgecolors='k')#, alpha=alpha_pm**4)
if len(inds_nm) != 0:
    plt.scatter(inputs_scatter[0, inds_nm], inputs_scatter[1, inds_nm], color = 'b', edgecolors='k')#, alpha=alpha_nm**4)
plt.title('Spirals task classification')
plt.show()




# ax.set_title("Experienced animal")
# plt.show()
