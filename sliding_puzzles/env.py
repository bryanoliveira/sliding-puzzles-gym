from enum import Enum
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image


# The 15-tile game environment
class SlidingEnv(gym.Env):
    metadata = {"render_modes": ["state", "human", "rgb_array"]}

    def __init__(
        self,
        w=4,
        h=None,
        shuffle_steps=100,
        shuffle_target_reward=None,
        render_mode="state",
        render_size=(32, 32),
        render_shuffling=False,
        sparse_rewards=False,
        win_reward=10,
        move_reward=0,
        invalid_move_reward=None,
        circular_actions=False,
        blank_value=-1,
        **kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.render_size = render_size
        if h is None:
            h = w
        self.grid_size_h = h
        self.grid_size_w = w
        assert shuffle_steps > 0, "A max shuffle step count must be set."
        self.shuffle_steps = shuffle_steps
        assert shuffle_target_reward is None or (
            shuffle_target_reward < 0 and shuffle_target_reward > -1
        ), "The target reward must be negative and greater than the theoretical minimum reward."
        self.shuffle_target_reward = shuffle_target_reward
        self.render_shuffling = render_shuffling
        self.sparse_rewards = sparse_rewards
        self.win_reward = win_reward
        self.move_reward = move_reward
        self.invalid_move_reward = invalid_move_reward
        self.circular_actions = circular_actions
        self.blank_value = blank_value

        # Define action and observation spaces
        self.observation_space = gym.spaces.Box(
            low=min(blank_value, 0), high=h * w, shape=(h, w), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(4)  # 4 actions (up, down, left, right)
        self.action_meanings = [
            "UP",  # moves the bottom piece up
            "DOWN",  # moves the top piece down
            "LEFT",  # moves the right piece to the left
            "RIGHT",  # moves the left piece to the right
        ]
        self.action = 4  # No action
        self.last_reward = self.move_reward
        self.last_done = False

        # Create an initial state with numbered tiles and one blank tile
        self.state = np.arange(0, h * w, dtype=np.int32).reshape((h, w))
        self.blank_pos = (0, 0)
        self.state[self.blank_pos] = self.blank_value

        # Initialize the plot
        if render_mode in ["human", "rgb_array"]:
            if render_mode == "rgb_array":
                plt.ioff()
            else:
                plt.ion()

            self.fig, self.ax = plt.subplots()

            # Setup app
            if render_mode == "human":

                def keypress(event):
                    if event.key in ["up", "down", "left", "right"]:
                        self.action = ["up", "down", "left", "right"].index(event.key)

                self.fig.canvas.manager.set_window_title("Sliding Block Puzzle")
                self.fig.canvas.mpl_connect("key_press_event", keypress)

            # Draw
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

    def step(self, action=None, force_dense_reward=False):
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
        }.get(action, (0, 0))  # 4: do nothing

        # Check if the move is valid (not out of bounds)
        if (
            0 <= y + dy < self.grid_size_h and 0 <= x + dx < self.grid_size_w
        ) or self.circular_actions:
            # Swap the blank tile with the adjacent tile
            # If the action is circular, swap the blank tile with the tile on the opposite side
            new_pos = (y + dy) % self.grid_size_h, (x + dx) % self.grid_size_w
            self.state[y, x], self.state[new_pos] = (
                self.state[new_pos],
                self.state[y, x],
            )
            self.blank_pos = new_pos
            reward, done = self.calculate_reward(force_dense=force_dense_reward)
        elif self.invalid_move_reward is not None:
            reward, done = self.invalid_move_reward, self.last_done
        else:
            reward, done = self.last_reward, self.last_done

        self.last_reward = reward
        self.last_done = done
        return self.state, reward, done, False, {"is_success": done}

    def reset(self, options=None, seed=None):
        # Create an initial state with numbered tiles and one blank tile
        self.state = np.arange(
            0, self.grid_size_h * self.grid_size_w, dtype=np.int32
        ).reshape((self.grid_size_h, self.grid_size_w))
        self.blank_pos = (0, 0)
        self.state[self.blank_pos] = self.blank_value

        # Shuffle the tiles
        self.shuffle(self.shuffle_steps)
        return self.state, {"is_success": False}

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
                img = np.array(self.fig.canvas.renderer._renderer)
                img = Image.fromarray(img)
                img = img.resize(self.render_size)
                return np.array(img)
        elif self.render_mode == "state":
            return self.state

    def close(self):
        if hasattr(self, "fig"):
            plt.close(self.fig)

    def __del__(self):
        self.close()

    def calculate_reward(self, force_dense=False):
        if np.all(self.state.flatten()[:-1] <= self.state.flatten()[1:]):
            return self.win_reward, True

        if not force_dense and self.sparse_rewards:
            return self.move_reward, False

        total_distance = 0
        for i in range(self.grid_size_h):
            for j in range(self.grid_size_w):
                value = self.state[i, j]
                if value > 0:
                    # Calculate goal position for the current value
                    goal_y, goal_x = divmod(value, self.grid_size_w)
                    # Sum the Manhattan distances
                    total_distance += abs(goal_y - i) + abs(goal_x - j)

        # Normalize the reward
        max_single_tile_distance = (self.grid_size_h - 1) + (self.grid_size_w - 1)
        max_distance = max_single_tile_distance * (
            self.grid_size_h * self.grid_size_w - 1
        )
        normalized_reward = -(total_distance / max_distance)

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
        r = 0

        while (
            # if target reward is not set, shuffle until max steps is reached
            self.shuffle_target_reward is None
            and steps > 0
        ) or (
            # if a target reward is set, shuffle until reach target or max steps is reached
            self.shuffle_target_reward is not None
            and r > self.shuffle_target_reward
            and steps > 0
        ):
            valid_actions = self.valid_actions()
            if undo_action in valid_actions:
                valid_actions.remove(undo_action)
            action = np.random.choice(valid_actions)
            undo_action = self.inverse_action(action)

            _, r, _, _, _ = self.step(action, force_dense_reward=True)

            if self.render_shuffling:
                self.render()

            steps -= 1

        if self.render_shuffling:
            print(f"Shuffling done! r={r} steps={steps}")


# Test the environment


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
