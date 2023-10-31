import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SlidingPuzzleEnv(gym.Env):
    def __init__(self, image_path, grid_size=3):
        super(SlidingPuzzleEnv, self).__init__()

        self.image = Image.open(image_path)
        self.image = self.image.resize((84, 84))
        self.grid_size = grid_size
        self.tile_size = self.image.width // grid_size
        self.n_tiles = grid_size * grid_size

        self.action_space = spaces.Discrete(4)  # Four possible actions: UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Box(0, 255, (self.image.width, self.image.height, 3), dtype=np.uint8)

        self.reset()

    def reset(self):
        self.state = np.array(self.image)
        self._shuffle()
        return self.state

    def step(self, action):
        if action == 0:  # UP
            self._move(0, -1)
        elif action == 1:  # DOWN
            self._move(0, 1)
        elif action == 2:  # LEFT
            self._move(-1, 0)
        elif action == 3:  # RIGHT
            self._move(1, 0)

        done = self._is_solved()
        reward = 0 if done else -1  # Reward is -1 for each step until the puzzle is solved.

        return self.state, reward, done, {}

    def render(self, mode='human'):
        plt.imshow(self.state)
        plt.show()

    def _shuffle(self):
        for _ in range(1000):  # You can adjust the number of shuffling steps
            valid_moves = self._valid_moves()
            random_action = self.np_random.choice(valid_moves)
            self.step(random_action)

    def _valid_moves(self):
        print("VALIIID", np.where(self.state[:, :, 0] == 255))
        i, j = np.where(self.state[:, :, 0] == 255)  # Find the position of the empty tile
        valid_moves = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT

        if i == 0:
            valid_moves.remove(0)
        if i == self.grid_size - 1:
            valid_moves.remove(1)
        if j == 0:
            valid_moves.remove(2)
        if j == self.grid_size - 1:
            valid_moves.remove(3)

        return valid_moves

    def _move(self, dx, dy):
        i, j = np.where(self.state[:, :, 0] == 255)  # Find the position of the empty tile
        new_i, new_j = i + dy, j + dx
        tile = self.state[new_i, new_j].copy()
        self.state[new_i, new_j] = self.state[i, j]
        self.state[i, j] = tile

    def _is_solved(self):
        return np.array_equal(self.state, np.array(self.image))

# Example usage:
if __name__ == '__main__':
    env = SlidingPuzzleEnv("image.jpg")
    obs = env.reset()

    for _ in range(100):  # Replace with your desired number of steps
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()  # Render the puzzle after each step
        if done:
            print("Puzzle Solved!")
            break
