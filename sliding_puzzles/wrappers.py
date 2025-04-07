import math
import os
import random
from typing import Optional

import gymnasium as gym
import numpy as np
from PIL import Image


class GymCompatibilityWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self._max_episode_steps = env.unwrapped.max_episode_steps

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation  # Return only observation for Gym compatibility

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info  # Convert to Gym style
    
    def render(self, *args, **kwargs):
        return self.env.render()


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

class CoordinatesWrapper(gym.ObservationWrapper):
    def __init__(self, env, normalize=True, **kwargs):
        super().__init__(env)
        self.normalize = (
            float(max(self.env.unwrapped.grid_size_h, self.env.unwrapped.grid_size_w) - 1)
            if normalize
            else 1.0
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.env.unwrapped.grid_size_h - 1,
            shape=(2 * self.env.unwrapped.grid_size_h * self.env.unwrapped.grid_size_w,),
            dtype=np.float32,
        )

    def observation(self, obs):
        # Create array of values, replacing negatives with 0
        goal_i, goal_j = np.divmod(obs.clip(min=0), self.env.unwrapped.grid_size_w)
        # Stack and flatten the coordinates
        coordinates = np.column_stack((goal_i.flatten(), goal_j.flatten())).flatten().astype(np.float32)
        if self.normalize:
            coordinates /= self.normalize
        return coordinates

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
        assert isinstance(image_size, tuple) or isinstance(image_size, int), f"{image_size}"
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
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
        self.current_image = None

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
        self.current_image = np.array(new_image, dtype=np.uint8)
        return self.current_image

    def render(self, mode=None):
        if mode is None:
            mode = self.env.unwrapped.render_mode

        if mode in ["human", "rgb_array"]:
            if mode == "rgb_array":
                return self.current_image

            self.env.unwrapped.ax.clear()
            self.env.unwrapped.ax.imshow(self.current_image)
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
        image_pool_size: Optional[int] = None,
        image_pool_seed: Optional[int] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if os.path.isabs(image_folder):
            self.image_folder = image_folder
        else:
            base_dir = os.path.dirname(__file__)
            self.image_folder = os.path.join(base_dir, "imgs", image_folder)

        all_images = os.listdir(self.image_folder)
        self.current_image_name = None

        if images is None:
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
        else:
            self.images = images

    def load_random_image(self):
        # load image
        self.current_image_name = random.choice(self.images)
        random_image_path = os.path.join(self.image_folder, self.current_image_name)
        image = Image.open(random_image_path).resize(self.image_size)
        return image


class ChannelFirstImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        assert len(env.observation_space.shape) == 3, f"{env.observation_space.shape}"
        assert env.observation_space.dtype == np.uint8, f"{env.observation_space.dtype}"
        channel_first_shape = tuple(sorted(env.observation_space.shape))
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


class ContinuousActionWrapper(gym.ActionWrapper):
    """
    Wrapper to convert continuous actions to discrete actions.
    Behaves like a joystick: first axis is up(+)/down(-), second is left(-)/right(+).
    If the activation threshold is met (which defaults to 0), the action with highest value will be selected.
    """

    def __init__(self, env, activation_threshold=0, **kwargs):
        super().__init__(env)
        self.activation_threshold = activation_threshold
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

    def action(self, action):
        """
        Converts a 2-axis continuous action into a discrete action between 0 and 4 (4 being a no-op).
        """
        if np.max(np.abs(action)) <= self.activation_threshold:
            return 4  # no-op

        if abs(action[0]) > abs(action[1]):
            return 0 if action[0] > 0 else 1  # up or down
        else:
            return 2 if action[1] < 0 else 3  # left or right
