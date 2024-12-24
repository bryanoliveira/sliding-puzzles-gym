import os
import random
import tarfile
from enum import Enum

import click
import gymnasium
import numpy as np
import requests

from sliding_puzzles import wrappers
from sliding_puzzles.env import SlidingEnv


class EnvType(Enum):
    raw = "raw"
    image = "image"
    normalized = "normalized"
    onehot = "onehot"


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
    
    if "continuous_actions" in env_config and env_config["continuous_actions"]:
        env = wrappers.ContinuousActionWrapper(env)

    return env


@click.group()
def cli():
    pass

@cli.group()
def setup():
    """Setup commands for different datasets."""
    pass

@setup.command()
def imagenet():
    """Download and extract images for the Sliding Puzzles environment."""
    url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/val_images.tar.gz"
    tar_file = "val_images.tar.gz"
    extract_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "imgs", "imagenet-1k")

    # Ask the user for their Hugging Face token
    token = os.getenv("HF_TOKEN")
    if token is None:
        token = click.prompt(f"Please enter your Hugging Face token to download the dataset from {url}", hide_input=True)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, stream=True, headers=headers)
    
    if response.status_code == 401:
        click.echo("Authentication failed. Please check your token and try again.")
        return

    total_size = int(response.headers.get('content-length', 0))
    
    with click.progressbar(length=total_size, label='Downloading images') as bar:
        with open(tar_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

    click.echo("Extracting images...")
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    os.remove(tar_file)
    click.echo(f"Images extracted to {extract_dir}")


gymnasium.envs.register(
    id="SlidingPuzzles-v0",
    entry_point=make,
)

def register_gym():
    import gym
    gym.envs.register(
        id="SlidingPuzzles-v0",
        entry_point=make,
    )

if __name__ == "__main__":
    cli()