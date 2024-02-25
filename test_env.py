import sliding_puzzles

env = sliding_puzzles.make(
    render_mode="human",
    w=3,
    h=3,
    sparse_rewards=True,
    move_reward=-1,
    shuffle_steps=100,
    # shuffle_target_reward=-0.5,
    # render_shuffling=True,
    variation="raw",
    # image_folder="imgs/single",
    # background_color_rgb=(255, 0, 0)
)
obs = env.reset()
total_reward = 0
while True:
    env.render()
    print(obs)
    obs, reward, done, trunc, info = env.step(None)
    total_reward += reward
    print(reward)
    if done or trunc:
        print("Done!", info)
        print("Total reward:", total_reward)
        total_reward = 0
        break


env.close()
