import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# The 15-tile game environment
class SlidingEnv(gym.Env):
    def __init__(self, w=4, h=4, shuffle_steps=100, render_shuffling=True):
        super().__init__()
        self.grid_size_h = h
        self.grid_size_w = w
        self.shuffle_steps = shuffle_steps
        self.render_shuffling = render_shuffling

        # Define action and observation spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=h * w, shape=(h, w), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(4)  # 4 actions (up, down, left, right)
        self.action_meanings = [
            "UP", # moves the bottom piece up
            "DOWN",  # moves the top piece down
            "LEFT",  # moves the right piece to the left
            "RIGHT",  # moves the left piece to the right
        ]

        # Create an initial state with numbered tiles and one blank tile
        self.state = np.arange(0, h * w).reshape((h, w))
        self.blank_pos = (0, 0)

        # Initialize the plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.mat = self.ax.matshow(
            np.zeros((h, w)), cmap=ListedColormap(["white", "gray"])
        )
        plt.yticks(range(h), [])
        plt.xticks(range(w), [])
        self.texts = [
            [
                self.ax.text(j, i, "", ha="center", va="center", fontsize=20)
                for j in range(w)
            ]
            for i in range(h)
        ]

    def step(self, action):
        # Get the position of the blank tile
        y, x = self.blank_pos

        # Define the action effects on the blank tile: (dy, dx)
        dy, dx = {
            0: (1, 0),  # Up: increase row index
            1: (-1, 0),  # Down: decrease row index
            2: (0, 1),  # Left: increase column index
            3: (0, -1),  # Right: decrease column index
        }.get(action, (0, 0))

        # Check if the move is valid (not out of bounds)
        if 0 <= y + dy < self.grid_size_h and 0 <= x + dx < self.grid_size_w:
            # Swap the blank tile with the adjacent tile
            self.state[y, x], self.state[y + dy, x + dx] = (
                self.state[y + dy, x + dx],
                self.state[y, x],
            )
            self.blank_pos = (y + dy, x + dx)
            reward = 0  # Example reward
        else:
            reward = -1  # Penalty for invalid move

        return self.state, reward, False, {}

    def reset(self):
        # Create an initial state with numbered tiles and one blank tile
        self.state = np.arange(0, self.grid_size_h * self.grid_size_w).reshape(
            (self.grid_size_h, self.grid_size_w)
        )
        self.blank_pos = (0, 0)
        self.shuffle(self.shuffle_steps)
        return self.state

    def render(self, mode="human"):
        self.mat.set_data(np.where(self.state > 0, 1, 0))  # Update the color data

        for i in range(self.grid_size_h):
            for j in range(self.grid_size_w):
                value = self.state[i, j]
                self.texts[i][j].set_text(
                    f"{value}" if value > 0 else ""
                )  # Update text

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def valid_actions(self):
        y, x = self.blank_pos
        valid_actions = []
        if y < self.grid_size_h - 1:
            # can move bottom tile up
            valid_actions.append(0)
        if y > 0:
            # can move top tile down
            valid_actions.append(1)
        if x < self.grid_size_w - 1:
            # can move right tile left
            valid_actions.append(2)
        if x > 0:
            # can move left tile right
            valid_actions.append(3)
        return valid_actions

    def shuffle(self, steps):
        if self.render_shuffling:
            print("Shuffling the puzzle...")
        for _ in range(steps):
            action = np.random.choice(self.valid_actions())
            self.step(action)
            if self.render_shuffling:
                self.render()
        if self.render_shuffling:
            print("Shuffling done!")


# Instantiate the environment
env = SlidingEnv()

# Test loop
for episode in range(10):  # Run 10 episodes for testing
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Select a random action
        observation, reward, done, info = env.step(action)  # Take a step
        env.render()  # Render the environment

        print("action:", env.action_meanings[action], "reward:", reward)
        time.sleep(1)

        if done:
            print(f"Episode {episode + 1} finished")
            break

env.close()
