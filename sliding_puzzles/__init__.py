import os
from enum import Enum
import shutil

import click
import gymnasium
import requests

from sliding_puzzles import wrappers
from sliding_puzzles.env import SlidingEnv


class EnvType(Enum):
    raw = "raw"
    image = "image"
    normalized = "normalized"
    onehot = "onehot"
    text_overlay = "text_overlay"
    coordinates = "coordinates"


def make(**env_config):
    seed = env_config.get("seed", None)
    if seed is not None:
        env_config["seed"] = seed

    if "w" not in env_config and "h" not in env_config:
        env_config["w"] = 3

    env = SlidingEnv(**env_config)

    if "variation" not in env_config or EnvType(env_config["variation"]) is EnvType.onehot:
        env = wrappers.OneHotEncodingWrapper(env)
    elif EnvType(env_config["variation"]) is EnvType.normalized:
        env = wrappers.NormalizedObsWrapper(env)
    elif EnvType(env_config["variation"]) is EnvType.coordinates:
        env = wrappers.CoordinatesWrapper(env)
    elif EnvType(env_config["variation"]) is EnvType.raw:
        pass
    elif EnvType(env_config["variation"]) is EnvType.image:
        assert "image_folder" in env_config, "image_folder must be specified in config"

        env = wrappers.ImageFolderWrapper(
            env,
            **env_config,
        )
    elif EnvType(env_config["variation"]) is EnvType.text_overlay:
        env = wrappers.TextOverlayWrapper(env, **env_config)
    
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
    extract_dir = os.path.join(os.path.dirname(__file__), "imgs", "imagenet-1k")

    # Check if dataset already exists
    if os.path.exists(extract_dir):
        click.echo("Skipping download since dataset already exists at " + extract_dir)
        return

    os.makedirs(extract_dir, exist_ok=True)

    token = os.getenv("HF_TOKEN")
    if token is None:
        token = click.prompt(
            f"Please enter your Hugging Face token to download the dataset from {url}",
            hide_input=True,
        )

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 401:
        click.echo("Authentication failed. Please check your token and try again.")
        return

    total_size = int(response.headers.get("content-length", 0))
    with click.progressbar(length=total_size, label="Downloading images") as bar:
        with open(tar_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

    click.echo("Extracting images...")
    shutil.unpack_archive(tar_file, extract_dir)
    os.remove(tar_file)
    click.echo(f"Images extracted to {extract_dir}")


@setup.command()
@click.option("-i", "--index", default=1, type=int, help="Start file index (default 1)")
@click.option("-r", "--range", "range_max", default=50, type=int, help="End file index (default 50)")
@click.option("-l", "--large", is_flag=True, default=False, help="Download from diffusiondb-large dataset")
def diffusiondb(index, range_max, large):
    """Download and extract images for the Sliding Puzzles environment from diffusiondb."""
    baseurl = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/"
    output = os.path.join(
        os.path.dirname(__file__),
        "imgs",
        "diffusiondb-large" if large else "diffusiondb",
    )

    if os.path.exists(output):
        click.echo("Skipping download since dataset already exists at " + output)
        return

    os.makedirs(output, exist_ok=True)

    with click.progressbar(range(index, range_max + 1), label="Downloading diffusiondb files") as bar:
        for idx in bar:
            if large:
                if idx <= 10000:
                    path = f"diffusiondb-large-part-1/part-{idx:06}.zip"
                else:
                    path = f"diffusiondb-large-part-2/part-{idx:06}.zip"
            else:
                path = f"images/part-{idx:06}.zip"
            url = baseurl + path
            file_path = os.path.join(output, os.path.basename(path))

            # Stream download with per-file progress bar
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with click.progressbar(length=total_size, label=f"Downloading {os.path.basename(path)}") as dl_bar:
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                dl_bar.update(len(chunk))
            except requests.exceptions.RequestException as e:
                click.echo(f"Error downloading {url} - {e}", err=True)
                continue

            try:
                shutil.unpack_archive(file_path, output)
            except Exception as e:
                click.echo(f"Error unzipping {file_path} - {e}", err=True)

            os.remove(file_path)


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