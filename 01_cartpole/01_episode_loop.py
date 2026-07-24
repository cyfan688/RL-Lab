import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()

done = False

step = 0

while not done:
    x, x_dot , theta, theta_dot = obs

    if theta > 0 :
        action = 1
    else:
        action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated

    step += 1

print("Episode Finished!")
print("Total Steps:",step)
env.close()