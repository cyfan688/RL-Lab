import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset()
action = env.action_space.sample()

next_obs, reward, terminated, truncated, info = env.step(action)

print("Action:")
print(action)

print()

print("Next Observation:")
print(next_obs)

print()

print("Reward:")
print(reward)

print()

print("Terminated:")
print(terminated)

print()

print("Truncated:")
print(truncated)

print("Info:")
print(info)
env.close()