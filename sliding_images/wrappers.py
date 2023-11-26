import gymnasium as gym
import numpy as np


class NormalizedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return observation / (self.grid_size_h * self.grid_size_w)
