import json
from src.evolution.Blueprint import get_connectivity_matrix
from matplotlib import pyplot as plt
import numpy as np

env_name = 'CDDM-v0'
max_timesteps = 3000
sleep = 0.01
filename = f"../../../data/evolved_models/{env_name}/CDDM-v0_generation=165_score=-0.0018402801366574505_N=1.json"
file = open(filename, "rb")
data = json.load(file)
with file as json_file:
    for line in json_file:
        data = json.loads(line)
genome_dict = data["genome dict"]
print(genome_dict)

W = get_connectivity_matrix(genome_dict)
print(W)

names = ["ctx m", "ctx c", "mr", "ml", "cr", "cl", "OutR", "OutL"]
n = len(names)
fig_w_rec = plt.figure()
ax = plt.gca()
im = ax.imshow(W, interpolation='blackman', cmap='coolwarm')
fig_w_rec.colorbar(im)

ax.set_xticks(np.arange(n))
ax.set_xticklabels(names)
ax.set_yticks(np.arange(n))
ax.set_yticklabels(names)

# Set ticks on both sides of axes on
ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
# Rotate and align bottom ticklabels
plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="right", va="center", rotation_mode="anchor")
# Rotate and align top ticklabels
plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="left", va="center",rotation_mode="anchor")

for (i,j), z in np.ndenumerate(W):
    if z < -0.05 or z > 0.05:
        if z >= -1:
            ax.text(j, i, str(np.round(z, 2)), ha="center", va="center", color='k')
        if z < -1:
            ax.text(j, i, str(np.round(z, 2)), ha="center", va="center", color='w')
# ax.set_title("Connectivity matrix", fontsize = 16, pad=10)
im = ax.imshow(W, interpolation='none', vmin=-np.max(np.abs(W)), vmax = np.max(np.abs(W)), cmap = 'coolwarm')
fig_w_rec.tight_layout()
plt.show()