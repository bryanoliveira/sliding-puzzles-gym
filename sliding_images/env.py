import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# The 15-tile game environment
class SlidingEnv(gym.Env):
    metadata = {"render_modes": ["state", "human", "rgb_array"]}

    def __init__(
        self, w=4, h=4, shuffle_steps=100, render_mode="state", render_shuffling=False
    ):
        super().__init__()
        self.render_mode = render_mode

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
            "UP",  # moves the bottom piece up
            "DOWN",  # moves the top piece down
            "LEFT",  # moves the right piece to the left
            "RIGHT",  # moves the left piece to the right
        ]
        self.action = 4  # No action

        # Create an initial state with numbered tiles and one blank tile
        self.state = np.arange(0, h * w, dtype=np.int32).reshape((h, w))
        self.blank_pos = (0, 0)

        # Initialize the plot
        def keypress(event):
            if event.key == "up":
                self.action = 0
            elif event.key == "down":
                self.action = 1
            elif event.key == "left":
                self.action = 2
            elif event.key == "right":
                self.action = 3

        if render_mode in ["human", "rgb_array"]:
            if render_mode == "rgb_array":
                plt.ioff()
            else:
                plt.ion()

            self.fig, self.ax = plt.subplots()
            self.fig.canvas.manager.set_window_title("Sliding Block Puzzle")
            self.fig.canvas.mpl_connect("key_press_event", keypress)
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

    def step(self, action=None):
        if action is None:
            action = self.action
        self.action = 4  # reset preset action

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
            reward, done = self.calculate_reward()
        else:
            reward = -1  # Penalty for invalid move
            done = False

        return self.state, reward, done, False, {}

    def reset(self, options=None, seed=None):
        # Create an initial state with numbered tiles and one blank tile
        self.state = np.arange(
            0, self.grid_size_h * self.grid_size_w, dtype=np.int32
        ).reshape((self.grid_size_h, self.grid_size_w))
        self.blank_pos = (0, 0)
        self.shuffle(self.shuffle_steps)
        return self.state, {}

    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            self.mat.set_data(np.where(self.state > 0, 1, 0))  # Update the color data

            for i in range(self.grid_size_h):
                for j in range(self.grid_size_w):
                    value = self.state[i, j]
                    self.texts[i][j].set_text(
                        f"{value}" if value > 0 else ""
                    )  # Update text

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            if self.render_mode == "rgb_array":
                return np.array(self.fig.canvas.renderer._renderer)
        elif self.render_mode == "state":
            return self.state

    def close(self):
        plt.close(self.fig)

    def calculate_reward(self):
        total_distance = 0
        solved = True
        for i in range(self.grid_size_h):
            for j in range(self.grid_size_w):
                value = self.state[i, j]
                if value != 0:
                    # Calculate goal position for the current value
                    goal_y, goal_x = divmod(value, self.grid_size_w)
                    # Sum the Manhattan distances
                    total_distance += abs(goal_y - i) + abs(goal_x - j)

        if total_distance == 0:
            return 10, True

        # Normalize the reward
        max_single_tile_distance = (self.grid_size_h - 1) + (self.grid_size_w - 1)
        max_distance = max_single_tile_distance * (
            self.grid_size_h * self.grid_size_w - 1
        )
        normalized_reward = 1 - (total_distance / max_distance)

        return normalized_reward, False

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

    def inverse_action(self, action):
        return {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
        }.get(action, 4)

    def shuffle(self, steps):
        if self.render_shuffling:
            print("Shuffling the puzzle...")
        undo_action = None
        s = None
        for _ in range(steps):
            valid_actions = self.valid_actions()
            if undo_action in valid_actions:
                valid_actions.remove(undo_action)
            action = np.random.choice(valid_actions)
            undo_action = self.inverse_action(action)

            _, r, _, _, _ = self.step(action)

            if self.render_shuffling:
                self.render()

        if self.render_shuffling:
            print("Shuffling done! r=", r)


if __name__ == "__main__":
    # Instantiate the environment
    env = SlidingEnv(render_mode="human")

    # Test loop
    for episode in range(10):  # Run 10 episodes for testing
        observation = env.reset()
        done = False
        while not done:
            env.render()  # Render the environment

            # action = np.random.choice(env.valid_actions())  # Choose a random action
            action = None
            observation, reward, done, trunc, info = env.step(action)  # Take a step

            print("reward:", reward)

            if done:
                print(f"Episode {episode + 1} finished")
                break

    env.close()
