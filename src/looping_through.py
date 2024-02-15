import numpy as np
from time import sleep
from src.slimevolleygym.slimevolley import SlimeVolleyEnv
np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True

if __name__=="__main__":

  # env = gym.make("SlimeVolley-v0")
  env = SlimeVolleyEnv()
  # policy1 = slimevolleygym.BaselinePolicy()  # defaults to use RNN Baseline for player
  # policy2 = RandomPolicy(None)  # defaults to use RNN Baseline for player

  if RENDER_MODE:
    env.seed(np.random.randint(0, 10000))
    env.reset()
    env.render()
  obs = env.reset()
  steps = 0
  total_reward = 0
  action = np.array([0, 0, 0])
  done = False
  while not done:

    # otherAction = np.array([0, 0, 0])
    # otherAction[np.random.randint(0, 2)] = 1
    # otherAction = policy2.predict(obs)

    action = np.zeros(3)
    otherAction = np.zeros(3)
    otherAction[np.random.randint(3)] = 1.0
    # env.game.agent_left.setAction(otherAction)
    # env.game.agent_right.setAction(otherAction)
    obs, reward, done, _ = env.step(action=action, otherAction=otherAction)
    total_reward += reward

    if RENDER_MODE:
      env.render()
      sleep(0.02) # 0.01

  env.close()
  print("cumulative score", total_reward)
