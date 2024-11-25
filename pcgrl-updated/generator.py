import os
from stable_baselines3 import PPO
import gymnasium as gym
import gym_pcgrl  # Ensure that the PCGRL environments are registered
from PIL import Image
import numpy as np
from model import FullyConvPolicySmallMap  # Import your custom policy if needed

# Load the trained model
model_path = 'runs/sokoban_54_log/best_model'  # Update this path to your actual model path
model = PPO.load(model_path)

# Define game and representation
game = 'sokoban'
representation = 'wide'  # Options: 'narrow', 'turtle', 'wide'

# Create environment name
env_name = f"{game}-{representation}-v0"

# Create the environment using the same parameters as during training
env = gym.make(env_name, rep=representation)

# For 'wide' representation, 'cropped_size' may not be relevant
# Verify observation space consistency
print(f"Environment observation space: {env.observation_space}")
print(f"Model's observation space: {model.observation_space}")

# Create a directory to save generated levels
os.makedirs('generated_levels', exist_ok=True)

# Generate levels
num_levels = 10  # Number of levels to generate
for i in range(num_levels):
    obs, info = env.reset()
    print(f"\nGenerating Level {i}:")
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    # Save the generated level
    img = env.render()
    img = Image.fromarray(img)
    img.save(f"generated_levels/level_{i}.png")
    # Optionally, print evaluation metrics
    print(f"Level {i} generated with final reward: {reward}")
    print(f"Info: {info}")
