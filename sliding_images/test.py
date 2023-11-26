import argparse
import time
import yaml

import gymnasium as gym
from stable_baselines3 import PPO

from env import SlidingEnv
import wrappers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model run id")
    args = parser.parse_args()

    with open(f"runs/{args.model}/configs.yaml", "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    gym.envs.register(
        id="SlidingEnv-v0",
        entry_point=SlidingEnv,
        max_episode_steps=configs["max_episode_steps"],
    )

    env = gym.make("SlidingEnv-v0", render_mode="human", **configs["env_kwargs"])
    env = configs["wrapper_class"](env)

    obs, info = env.reset()
    terminated = False
    truncated = False

    model = PPO.load(f"runs/{args.model}/model", env=env)

    while not (terminated or truncated):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action.item())
        print(terminated, truncated, reward)
        env.render()
        time.sleep(0.5)

    time.sleep(60)
    env.close()