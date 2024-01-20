from enum import Enum

from sliding_puzzles.env import SlidingEnv
from sliding_puzzles import wrappers


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

        env = ImagePuzzleWrapper(
            env,
            **env_config,
        )
    return env
