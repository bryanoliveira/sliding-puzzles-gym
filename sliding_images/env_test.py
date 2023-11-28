import gymnasium as gym

from env import SlidingEnv
import wrappers

env = SlidingEnv(w=2, h=2, render_mode="human", sparse_rewards=True)
# env = wrappers.ImagePuzzleWrapper(env)
env.reset()

while True:
    env.render()
    obs, reward, done, trunc, info = env.step(None)
    print(reward)
    if done or trunc:
        print("Done!")
        break

env.close()