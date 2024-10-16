import numpy as np
import gymnasium as gym
import sliding_puzzles

env = gym.make(
    "SlidingPuzzles-v0",
    seed=2,
    render_mode="human",
    w=3,
    # circular_actions=True,

    # sparse_rewards=True,
    # sparse_mode="win",
    # win_reward=10,
    # move_reward=-1,
    # invalid_move_reward=-10,
    # reward_mode = "percent_solved",

    # shuffle_target_reward=-0.7,
    # shuffle_render=True,
    # shuffle_mode="serial",
    # shuffle_steps=0,

    variation=["raw", "onehot", "image"][2],
    image_folder=[
        "test",
        "imagenet-1k",
    ][1],
    image_pool_size=3,
    # background_color_rgb=(255, 0, 0)
    image_size=(1024, 1024),
)
env.unwrapped.setup_render_controls()
obs, info = env.reset()
print(info)
total_reward = 0
while True:
    env.render()
    # action = np.random.choice(env.valid_actions())
    action = None
    obs, reward, done, trunc, info = env.step(action)
    if info["last_action"] < 4:
        total_reward += reward
        print(reward, done, trunc, info, total_reward)
        if done or trunc:
            env.render()
            total_reward = 0
            print("Done!", info)
            print("Total reward:", total_reward)
            print("Press r to reset")
