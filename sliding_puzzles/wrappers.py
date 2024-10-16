import math
import os
import random

import gymnasium as gym
import numpy as np
from PIL import Image


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
                    # blank tile may be negative
                    + (tile_value if tile_value > 0 else 0)
                ] = 1
        return one_hot_encoded


class ExponentialRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)

    def reward(self, reward):
        return np.exp(reward)


class BaseImageWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        image_size=(84, 84),  # width x height
        background_color_rgb=(0, 0, 0),
        **kwargs,
    ):
        super().__init__(env)
        self.image_size = image_size
        self.background_color_rgb = background_color_rgb
        self.section_size = (
            math.ceil(image_size[0] / self.env.unwrapped.grid_size_w),
            math.ceil(image_size[1] / self.env.unwrapped.grid_size_h),
        )
        self.image_sections = []
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=tuple(self.image_size[::-1]) + (3,),  # height x width channels
            dtype=np.uint8,
        )
        self.env.unwrapped.render = self.render

    def reset(self, **kwargs):
        image = self.load_random_image()
        self.set_split_image(image)
        return super().reset(**kwargs)

    def load_random_image(self):
        raise NotImplementedError

    def set_split_image(self, image):
        # split image
        self.image_sections = []
        for i in range(self.env.unwrapped.grid_size_h):
            for j in range(self.env.unwrapped.grid_size_w):
                left = j * self.section_size[0]
                upper = i * self.section_size[1]
                right = min(left + self.section_size[0], self.image_size[0])
                lower = min(upper + self.section_size[1], self.image_size[1])
                section = image.crop((left, upper, right, lower))
                section = section.resize(self.section_size)
                self.image_sections.append(section)

    def observation(self, obs):
        new_image = Image.new("RGB", self.image_size, self.background_color_rgb)
        # paint tiles
        for i in range(self.env.unwrapped.grid_size_h):
            for j in range(self.env.unwrapped.grid_size_w):
                section_idx = obs[i, j]
                if section_idx > 0:
                    section = self.image_sections[section_idx - 1]
                    new_image.paste(
                        section, (j * self.section_size[0], i * self.section_size[1])
                    )

        return np.array(new_image, dtype=np.uint8)

    def render(self, mode=None):
        if mode is None:
            mode = self.env.unwrapped.render_mode

        if mode in ["human", "rgb_array"]:
            current_obs = self.env.unwrapped.state
            img_obs = self.observation(current_obs)

            if mode == "rgb_array":
                return img_obs

            self.env.unwrapped.ax.clear()
            self.env.unwrapped.ax.imshow(img_obs)
            self.env.unwrapped.fig.canvas.draw()
            self.env.unwrapped.fig.canvas.flush_events()

        elif mode == "state":
            return self.env.unwrapped.state

    def setup_render_controls(self):
        # required so subclass self.reset function can be called from keypress
        self.env.unwrapped.setup_render_controls(self)


class ImageFolderWrapper(BaseImageWrapper):
    def __init__(
        self,
        env,
        image_folder: str = "single",
        image_pool_size: int = None,
        image_pool_seed: int = None,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if os.path.isabs(image_folder):
            self.image_folder = image_folder
        else:
            base_dir = os.path.dirname(__file__)
            self.image_folder = os.path.join(base_dir, "../imgs", image_folder)

        all_images = os.listdir(self.image_folder)
        if image_pool_size is None:
            image_pool_size = 1 if image_folder == "single" else len(all_images)
            print(
                f"Inferring image pool size from folder {image_folder}: {image_pool_size}"
            )
        else:
            image_pool_size = int(image_pool_size)

        if image_pool_size > len(all_images):
            print(
                f"WARNING: Image pool size {image_pool_size} is greater "
                f"than the number of images in the folder {self.image_folder}. "
                f"Reducing to {len(all_images)}"
            )
            image_pool_size = len(all_images)

        if image_pool_seed is None:
            image_pool_seed = self.env.unwrapped.seed

        rng = random.Random(image_pool_seed)
        self.images = rng.sample(all_images, int(image_pool_size))

    def load_random_image(self):
        # load image
        random_image_path = os.path.join(self.image_folder, random.choice(self.images))
        image = Image.open(random_image_path).resize(self.image_size)
        return image


class ChannelFirstImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        assert len(env.observation_space.shape) == 3, f"{env.observation_space.shape}"
        assert env.observation_space.dtype == np.uint8, f"{env.observation_space.dtype}"
        channel_first_shape = sorted(env.observation_space.shape)
        self.should_transpose = channel_first_shape != env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=channel_first_shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        if self.should_transpose:
            return obs.transpose(2, 0, 1)
        return obs


class NormalizedImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32) / 255.0
