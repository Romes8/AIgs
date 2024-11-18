import gym
import gym_sokoban

# Print gym version to confirm
print(f"Gym version: {gym.__version__}")

# Create the Sokoban environment
env = gym.make('Sokoban-v0')

# Reset the environment
observation = env.reset()

# Render the environment
env.render(mode='human')  # Visual rendering

# Take a random action
action = env.action_space.sample()  # Random action
observation, reward, done, info = env.step(action)

# Print results
print(f"Action: {action}")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info: {info}")

# Close the environment
env.close()
