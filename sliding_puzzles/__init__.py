from enum import Enum
import gymnasium as gym
from sliding_puzzles.env import SlidingEnv
from sliding_puzzles import wrappers


MAX_EPISODE_STEPS=1000

class EnvType(Enum):
    image = "image"
    normalized = "normalized"
    onehot = "onehot"


def make(**env_config):
    env = SlidingEnv(**env_config)

    if (
        "variation" not in env_config
        or EnvType(env_config["variation"]) is EnvType.normalized
    ):
        pass
    elif EnvType(env_config["variation"]) is EnvType.onehot:
        env = wrappers.OneHotEncodingWrapper(env)
    elif EnvType(env_config["variation"]) is EnvType.image:
        assert "image_folder" in env_config, "image_folder must be specified in config"

        env = wrappers.ImagePuzzleWrapper(
            env,
            **env_config,
        )
    return env


gym.envs.register(
    id="SlidingPuzzle-v0",
    entry_point=make,
    max_episode_steps=MAX_EPISODE_STEPS,
)