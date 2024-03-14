import sliding_puzzles

env = sliding_puzzles.make(
    render_mode="human",
    w=2,
    # circular_actions=True,

    # sparse_rewards=True,
    # sparse_mode="win",
    win_reward=10,
    # move_reward=-1,
    # invalid_move_reward=-10,

    variation=["raw", "onehot", "image"][2],
    image_folder=[
        "/mnt/data/Documents/sliding-puzzle/imgs/single",
        "/mnt/data/Documents/sliding-puzzle/imgs/mnist",
        "/mnt/data/Documents/sliding-puzzle/imgs/imagenet-1k",
    ][2],
    # background_color_rgb=(255, 0, 0)
    # image_size=(210, 160),
)
env.setup_render_controls()
obs, info = env.reset()
print(info)
total_reward = 0
while True:
    env.render()
    obs, reward, done, trunc, info = env.step(None)
    if info["last_action"] < 4:
        total_reward += reward
        print(reward, done, trunc, info, total_reward)
        if done or trunc:
            env.render()
            total_reward = 0
            print("Done!", info)
            print("Total reward:", total_reward)
            print("Press r to reset")

env.close()
