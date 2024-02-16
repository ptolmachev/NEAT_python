import gymnasium
import numpy as np
from matplotlib import pyplot as plt
from src.environments.CDDM_environment import CDDM_environment

# Register the environment
gymnasium.register(
    id='CDDM-v0',
    entry_point='CDDM_environment:CDDM_environment',
    kwargs={'n_steps': 300}
)

env = gymnasium.make('CDDM-v0', n_steps = 300)
obs = env.reset()
done = False
action_sequence = []
mse = 0
while not done:
    rel_inds = [2, 3] if (obs[0] == 1) else [4, 5]
    rel_info = np.take(obs, rel_inds)
    action = np.zeros(2)
    ind = 0 if (rel_info[0] > rel_info[1]) else 1
    action[ind] = 1
    action += 0.01 * np.random.randn()
    obs, reward, done, _ = env.step(action)
    mse += reward
    action_sequence.append(action)

print(f"Total MSE: {mse}")

fig = plt.figure()
plt.plot(action_sequence, color='r')
plt.grid(True)
plt.show()

