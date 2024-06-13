from enum import Enum
import random

import gymnasium as gym
import numpy as np
from sliding_puzzles.env import SlidingEnv
from sliding_puzzles import wrappers


class EnvType(Enum):
    raw = "raw"
    image = "image"
    normalized = "normalized"
    onehot = "onehot"
    imagenet = "imagenet"


def make(**env_config):
    seed = env_config.get("seed", None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    env = SlidingEnv(**env_config)

    if "variation" not in env_config or EnvType(env_config["variation"]) is EnvType.raw:
        pass
    elif EnvType(env_config["variation"]) is EnvType.normalized:
        env = wrappers.NormalizedObsWrapper(env)
    elif EnvType(env_config["variation"]) is EnvType.onehot:
        env = wrappers.OneHotEncodingWrapper(env)
    elif EnvType(env_config["variation"]) is EnvType.image:
        assert "image_folder" in env_config, "image_folder must be specified in config"

        env = wrappers.ImageFolderWrapper(
            env,
            **env_config,
        )
    elif EnvType(env_config["variation"]) is EnvType.imagenet:
        env = wrappers.ImagenetWrapper(
            env,
            **env_config,
        )
    return env


gym.envs.register(
    id="SlidingPuzzle-v0",
    entry_point=make,
)