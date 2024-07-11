import random
from typing import Optional

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from sliding_puzzles.utils import count_inversions, is_solvable, inverse_action


# The 15-tile game environment
class SlidingEnv(gym.Env):
    metadata = {
        "render_modes": ["state", "human", "rgb_array"],
        "render.modes": ["state", "human", "rgb_array"],
        "reward_modes": ["distances", "percent_solved"],
        "shuffle_modes": ["fast", "serial"],
        "render_fps": 10,
    }

    def __init__(
        self,
        w: Optional[int] = None,
        h: Optional[int] = None,
        render_mode: str = "state",
        render_size: tuple = (32, 32),  # W x H
        sparse_rewards: bool = False,
        win_reward: float = 10,
        move_reward: float = -0.1,
        invalid_move_reward: Optional[float] = -1,
        circular_actions: bool = False,
        blank_value: int = -1,
        reward_mode: str = "distances",
        shuffle_mode: str = "fast",
        shuffle_steps: int = 100,
        shuffle_target_reward: Optional[float] = None,
        shuffle_render: bool = False,
        max_episode_steps: Optional[int] = 1000,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        # Config
        assert (
            render_mode in self.metadata["render_modes"]
        ), f"render_mode must be one of {self.metadata['render_modes']}. Got: {render_mode}"
        self.render_mode = render_mode
        assert (
            render_size is not None and len(render_size) == 2
        ), f"Render size must have len=2. Got: {render_size}"
        self.render_size = render_size
        assert w or h, f"At least one of the grid dimensions must be set. Got {w}x{h}"
        if h is None:
            h = w
        elif w is None:
            w = h
        assert (
            w > 1 or h > 1
        ), f"At least one of the grid dimensions must be greater than 1. Got {w}x{h}"
        self.grid_size_h = h
        self.grid_size_w = w
        self.sparse_rewards = sparse_rewards
        assert (
            win_reward != move_reward
        ), f"win_reward must be different from the move reward. Got: win_reward={win_reward}, move_reward={move_reward}"
        assert (
            type(win_reward) in [int, float] and win_reward >= 0
        ), f"win_reward must be numeric and not negative. Got: {win_reward} (type: {type(win_reward)})"
        self.win_reward = win_reward
        assert (
            invalid_move_reward != move_reward
        ), f"invalid_move_reward must be None or different from the move reward. Got: invalid_move_reward={invalid_move_reward}, move_reward={move_reward}"
        assert (
            type(move_reward) in [int, float] and move_reward <= 0
        ), f"move_reward must be numeric and not positive. Got: {move_reward} (type: {type(move_reward)})"
        self.move_reward = move_reward
        if invalid_move_reward is not None:
            assert (
                type(invalid_move_reward)
                in [
                    int,
                    float,
                ]
                and invalid_move_reward < 0
            ), f"invalid_move_reward must be None or numeric and not negative. Got: {invalid_move_reward} (type: {type(invalid_move_reward)})"
        self.invalid_move_reward = invalid_move_reward
        assert (
            reward_mode in self.metadata["reward_modes"]
        ), f"reward_mode must be one of {self.metadata['reward_modes']}. Got: {reward_mode}"
        self.reward_mode = reward_mode
        self.circular_actions = circular_actions
        assert (
            type(blank_value) is int and blank_value <= 0
        ), f"blank_value must be a non-positive integer. Got {blank_value} (type: {type(blank_value)})"
        self.blank_value = blank_value
        assert (
            shuffle_mode in self.metadata["shuffle_modes"]
        ), f"shuffle_mode must be one of {self.metadata['shuffle_modes']}. Got: {shuffle_mode}"
        self.shuffle_mode = shuffle_mode
        assert type(shuffle_steps) is int and (
            shuffle_mode == "fast" or shuffle_steps >= 0
        ), f"shuffle_steps must be a non-zero integer. Got: {shuffle_steps} (type: {type(shuffle_steps)})"
        self.shuffle_steps = shuffle_steps
        assert shuffle_target_reward is None or (
            type(shuffle_target_reward) is float and shuffle_target_reward < 0
        ), f"shuffle_target_reward must be None or a negative float. Got: {shuffle_target_reward} (type: {type(shuffle_target_reward)})"
        self.shuffle_target_reward = shuffle_target_reward
        self.shuffle_render = shuffle_render
        assert (
            max_episode_steps is None or max_episode_steps > 0
        ), f"max_episode_steps must be a positive integer or None. Got: {max_episode_steps} (type: {type(max_episode_steps)})"
        self.max_episode_steps = max_episode_steps
        self.seed = int(seed) if seed else random.randint(1, 1000000)

        # Define action and observation spaces
        self.observation_space = gym.spaces.Box(
            low=min(blank_value, 0), high=h * w, shape=(h, w), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(4)
        self.action_meanings = [
            "UP",  # moves the bottom piece up
            "DOWN",  # moves the top piece down
            "LEFT",  # moves the right piece to the left
            "RIGHT",  # moves the left piece to the right
        ]

        # Initializations
        self.action = 4  # No action
        self.last_reward = self.move_reward
        self.last_terminated = False
        self.steps = 0

        # Create an initial state with numbered tiles and one blank tile
        self.set_solved_puzzle()

        # Calculate the max distance each tile can be from its goal
        self.total_max_distances = 0
        for goal_y in range(self.grid_size_h):
            for goal_x in range(self.grid_size_w):
                if (goal_y, goal_x) == (self.grid_size_h - 1, self.grid_size_w - 1):
                    # skip blank tile
                    continue

                self.total_max_distances += max(
                    goal_y, (self.grid_size_h - 1) - goal_y
                ) + max(goal_x, (self.grid_size_w - 1) - goal_x)

        # Initialize the plot
        if render_mode in ["human", "rgb_array"]:
            if render_mode == "rgb_array":
                plt.ioff()
            else:
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
            if render_mode == "human":
                self.fig.canvas.manager.set_window_title("Sliding Block Puzzle")

    def step(self, action=None, force_dense_reward=False):
        if action is None:
            action = self.action
        self.action = 4  # reset preset action to "do nothing"

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
            reward, terminated = self.calculate_reward(force_dense=force_dense_reward)
        elif self.invalid_move_reward is not None:
            reward, terminated = self.invalid_move_reward, self.last_terminated
        else:
            reward, terminated = self.last_reward, self.last_terminated

        self.last_reward = reward
        self.last_terminated = terminated
        if not force_dense_reward:
            self.steps += 1
        
        truncated = self.max_episode_steps and self.steps >= self.max_episode_steps
        info = {"is_success": terminated, "state": self.state, "last_action": action}

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, info

    def reset(self, options=None, seed=None):
        self.steps = 0
        # Create an initial state with numbered tiles and one blank tile
        if self.shuffle_mode == "fast":
            self.set_shuffled_puzzle()
        else:
            self.set_solved_puzzle()
            self.shuffle_serial(
                steps=self.shuffle_steps,
                target_reward=self.shuffle_target_reward,
                render=self.shuffle_render,
            )

        if self.render_mode == "human":
            self.render()

        return self.state, {"is_success": False, "state": self.state}

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode

        if mode in ["human", "rgb_array"]:
            # Update the color data
            self.mat.set_data(np.where(self.state > 0, 1, 0))

            for i in range(self.grid_size_h):
                for j in range(self.grid_size_w):
                    value = self.state[i, j]
                    self.texts[i][j].set_text(
                        f"{value}" if value > 0 else ""
                    )  # Update text

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            if mode == "rgb_array":
                img = np.array(self.fig.canvas.renderer._renderer)
                img = Image.fromarray(img).convert("RGB")
                img = img.resize(self.render_size)
                return np.array(img, dtype=np.uint8)
        elif mode == "state":
            return self.state

    def setup_render_controls(self, env_instance=None):
        action_keys = [a.lower() for a in self.action_meanings]

        def keypress(event):
            if event.key in action_keys:
                self.action = action_keys.index(event.key)
            elif event.key == "r":
                if env_instance is not None:
                    env_instance.reset()
                else:
                    self.reset()

        self.fig.canvas.mpl_connect("key_press_event", keypress)

    def close(self):
        if hasattr(self, "fig") and self.fig is not None:
            try:
                plt.close(self.fig)
            except:
                pass

    def __del__(self):
        self.close()

    def calculate_reward(self, force_dense=False):
        # Considering the blank value is always less than any other value,
        # we can check if the puzzle is solved by checking if the state is sorted
        flat_state = self.state.flatten()
        if flat_state[-1] == self.blank_value and np.all(
            flat_state[:-2] <= flat_state[1:-1]
        ):
            return self.win_reward, True

        if not force_dense and self.sparse_rewards:
            return self.move_reward, False

        if self.reward_mode == "distances":
            total_distance = 0
            for i in range(self.grid_size_h):
                for j in range(self.grid_size_w):
                    value = self.state[i, j]
                    if value > 0:
                        # Calculate goal position for the current value
                        goal_y, goal_x = divmod(value - 1, self.grid_size_w)
                        # Sum the Manhattan distances
                        total_distance += abs(goal_y - i) + abs(goal_x - j)

            # Normalize the reward
            reward = -(total_distance / self.total_max_distances)
        else:
            solved = np.arange(1, self.grid_size_h * self.grid_size_w)
            reward = -np.mean(flat_state[:-1] != solved)

        return reward, False

    def set_solved_puzzle(self):
        self.state = np.arange(1, self.grid_size_h * self.grid_size_w + 1).reshape(
            self.grid_size_h, self.grid_size_w
        )
        self.blank_pos = (self.grid_size_h - 1, self.grid_size_w - 1)
        self.state[self.blank_pos] = self.blank_value

    def set_shuffled_puzzle(self):
        # Exclude the blank tile for shuffling
        puzzle_array = np.arange(1, self.grid_size_h * self.grid_size_w)
        # Shuffle the array
        np.random.shuffle(puzzle_array)
        inversions = count_inversions(puzzle_array)
        # Randomly choose a row for the blank tile
        self.blank_pos = (
            random.randint(0, self.grid_size_h - 1),
            random.randint(0, self.grid_size_w - 1),
        )

        # Adjust the puzzle to make sure it's solvable
        if not is_solvable(
            inversions, self.blank_pos[0], self.grid_size_w, self.grid_size_h
        ):
            # Swap the first two tiles
            puzzle_array[0], puzzle_array[1] = puzzle_array[1], puzzle_array[0]
            # Recalculate inversions after swap
            inversions = count_inversions(puzzle_array)
            assert is_solvable(
                inversions, self.blank_pos[0], self.grid_size_w, self.grid_size_h
            ), "Shuffled puzzle is not solvable!"

        # Place the blank tile in the puzzle
        puzzle_array = np.insert(
            puzzle_array,
            self.grid_size_w * self.blank_pos[0] + self.blank_pos[1],
            self.blank_value,
        )
        self.state = puzzle_array.reshape((self.grid_size_h, self.grid_size_w))

        # If the puzzle is solved, execute a random action
        if self.calculate_reward()[0] == self.win_reward:
            self.step(np.random.choice(self.valid_actions()), force_dense_reward=True)

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

    def shuffle_serial(self, steps=1000, render=False, target_reward=None):
        if render:
            print("Shuffling the puzzle...")

        if steps == 0:
            return

        undo_action = None
        r = 0
        while (
            (
                # if target reward is not set, shuffle until max steps is reached
                target_reward is None
                and steps > 0
            )
            or (
                # if a target reward is set, shuffle until reach target or max steps is reached
                target_reward is not None
                and r > target_reward
                and steps > 0
            )
            or (
                # continue shuffling until the puzzle is not solved
                r
                == self.win_reward
            )
        ):
            valid_actions = self.valid_actions()
            if undo_action in valid_actions:
                valid_actions.remove(undo_action)
            action = np.random.choice(valid_actions)
            undo_action = inverse_action(action)

            _, r, _, _, _ = self.step(action, force_dense_reward=True)

            if render:
                self.render()

            steps -= 1

        if render:
            print(f"Shuffling done! r={r} steps={steps}")


# Test the environment


if __name__ == "__main__":
    # Instantiate the environment
    env = SlidingEnv(render_mode="human")

    # Test loop
    for episode in range(10):  # Run 10 episodes for testing
        observation, info = env.reset()
        done = False
        while not done:
            env.render()  # Render the environment

            # action = np.random.choice(env.valid_actions())  # Choose a random action
            action = None
            observation, reward, terminated, truncated, info = env.step(action)  # Take a step
            done = terminated or truncated
            if info["last_action"] < 4:
                print("reward:", reward)

            if done:
                print(f"Episode {episode + 1} finished")
                break

    env.close()
