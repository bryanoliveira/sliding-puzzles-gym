import gymnasium as gym
import numpy as np
from PIL import Image
import os
import random


class NormalizedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=self.env.unwrapped.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        # divide where > 0 by grid size,
        # leave < 0 untouched.
        return np.where(
            observation > 0,
            observation
            / (self.env.unwrapped.grid_size_h * self.env.unwrapped.grid_size_w - 1),
            observation,
        ).astype(np.float32)


class OneHotEncodingWrapper(gym.ObservationWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                self.env.unwrapped.grid_size_h
                * self.env.unwrapped.grid_size_w
                * self.env.unwrapped.grid_size_h
                * self.env.unwrapped.grid_size_w,
            ),
            dtype=np.float32,
        )

    def observation(self, obs):
        one_hot_encoded = np.zeros(self.observation_space.shape, dtype=np.float32)
        for i in range(self.env.unwrapped.grid_size_h):
            for j in range(self.env.unwrapped.grid_size_w):
                tile_value = obs[i, j]
                one_hot_index = i * self.env.unwrapped.grid_size_w + j
                one_hot_encoded[
                    one_hot_index
                    * self.env.unwrapped.grid_size_h
                    * self.env.unwrapped.grid_size_w
                    + (tile_value if tile_value > 0 else 0)  # blank tile may be negative
                ] = 1
        return one_hot_encoded


class ImagePuzzleWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        image_folder="img",
        image_size=(128, 128),  # width x height
        background_color_rgb=(0, 0, 0),
        normalize=True,
        **kwargs
    ):
        super().__init__(env)
        self.image_folder = image_folder
        self.image_size = image_size
        self.background_color_rgb = background_color_rgb
        self.normalize = normalize
        self.section_size = (
            image_size[1] // self.env.unwrapped.grid_size_h,
            image_size[0] // self.env.unwrapped.grid_size_w,
        )
        self.image_sections = []
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=tuple(self.image_size[::-1]) + (3,),  # height x width channels
            dtype=np.float32 if self.normalize else np.uint8,
        )

    def reset(self, **kwargs):
        self.load_random_image()
        return super().reset(**kwargs)

    def load_random_image(self):
        # load image
        images = os.listdir(self.image_folder)
        random_image_path = os.path.join(self.image_folder, random.choice(images))
        image = Image.open(random_image_path).resize(self.image_size)

        # split image
        self.image_sections = []
        for i in range(self.env.unwrapped.grid_size_h):
            for j in range(self.env.unwrapped.grid_size_w):
                left = j * self.section_size[1]
                upper = i * self.section_size[0]
                right = left + self.section_size[1]
                lower = upper + self.section_size[0]
                section = image.crop((left, upper, right, lower))
                self.image_sections.append(section)

    def observation(self, obs, skip_normalization=False):
        new_image = Image.new("RGB", self.image_size, self.background_color_rgb)
        # paint tiles
        for i in range(self.env.unwrapped.grid_size_h):
            for j in range(self.env.unwrapped.grid_size_w):
                section_idx = obs[i, j]
                if section_idx > 0:
                    section = self.image_sections[section_idx - 1]
                    new_image.paste(
                        section, (j * self.section_size[1], i * self.section_size[0])
                    )

        if not skip_normalization and self.normalize:
            return np.array(new_image, dtype=np.float32) / 255

        return np.array(new_image, dtype=np.uint8)

    def render(self, mode="human"):
        if self.env.unwrapped.render_mode in ["human", "rgb_array"]:
            current_obs = self.env.unwrapped.state
            img_obs = self.observation(current_obs, skip_normalization=True)

            if self.env.unwrapped.render_mode == "rgb_array":
                return img_obs

            self.env.unwrapped.ax.clear()
            self.env.unwrapped.ax.imshow(img_obs)
            self.env.unwrapped.fig.canvas.draw()
            self.env.unwrapped.fig.canvas.flush_events()

        elif self.env.unwrapped.render_mode == "state":
            return self.env.unwrapped.state


class ExponentialRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)

    def reward(self, reward):
        return np.exp(reward)
