import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Placeholder for the 15-tile game environment
class PlaceholderEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # Example: 4 actions (up, down, left, right)
        self.observation_space = gym.spaces.Box(low=0, high=15, shape=(4, 4), dtype=np.int32)

        self.state = np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15,  0]
        ])  # Initial state

    def step(self, action):
        # Implement the step function (to be done later)
        return self.state, 0, False, {}

    def reset(self):
        # Implement the reset function (to be done later)
        return self.state

    def render(self, mode='human'):
        fig, ax = plt.subplots()
        ax.matshow(np.where(self.state > 0, 1, 0), cmap=ListedColormap(['white', 'gray']))

        for (i, j), value in np.ndenumerate(self.state):
            ax.text(j, i, f'{value}' if value > 0 else '', ha='center', va='center', fontsize=20)

        plt.xticks([])
        plt.yticks([])
        plt.show()

# Instantiate the environment
env = PlaceholderEnv()

# Test loop
for episode in range(10):  # Run 10 episodes for testing
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Select a random action
        print("step", action)
        observation, reward, done, info = env.step(action)  # Take a step
        env.render()  # Render the environment

        if done:
            print(f"Episode {episode + 1} finished")
            break

env.close()