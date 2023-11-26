import gymnasium as gym
import numpy as np


class NormalizedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return observation / (self.unwrapped.grid_size_h * self.unwrapped.grid_size_w)


class OneHotEncodingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(OneHotEncodingWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                env.grid_size_h * env.grid_size_w * env.grid_size_h * env.grid_size_w,
            ),
            dtype=np.float32,
        )

    def observation(self, obs):
        one_hot_encoded = np.zeros(self.observation_space.shape)
        for i in range(self.unwrapped.grid_size_h):
            for j in range(self.unwrapped.grid_size_w):
                tile_value = obs[i, j]
                one_hot_index = i * self.unwrapped.grid_size_w + j
                one_hot_encoded[
                    one_hot_index
                    * self.unwrapped.grid_size_h
                    * self.unwrapped.grid_size_w
                    + tile_value
                ] = 1
        return one_hot_encoded
