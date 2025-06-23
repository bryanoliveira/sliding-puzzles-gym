import numpy as np
import gymnasium as gym
import sliding_puzzles

env = gym.make(
    "SlidingPuzzles-v0",
    # seed=2,
    render_mode="human",
    w=3,

    # -- game rules
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

    variation=["raw", "onehot", "image", "text_overlay"][3],

    # -- for image variation
    # image_folder=[
    #     "demo",
    #     "imagenet-1k",
    # ][0],
    # image_pool_size=3,
    # # background_color_rgb=(255, 0, 0)
    # image_size=(1024, 1024),
    # images=[
    #     "poodle.jpeg"
    # ],

    # -- for text variation
    texts=[
        "The quick brown fox jumps over the lazy dog.",
    ],
    h_separator=" ",
)
env.unwrapped.setup_render_controls()

state = [
    [-1, 5, 2],
    [7, 3, 4],
    [8, 6, 1]
]
obs, info = env.reset(options={"state": state})
print(obs, info)

total_reward = 0
while True:
    env.render()
    # action = np.random.choice(env.unwrapped.valid_actions())
    action = None
    obs, reward, done, trunc, info = env.step(action)
    if info["last_action"] < 4:
        total_reward += reward
        print(obs)
        print(reward, done, trunc, info, total_reward)
        if done or trunc:
            env.render()
            total_reward = 0
            print("Done!", info)
            print("Total reward:", total_reward)
            print("Press r to reset")
