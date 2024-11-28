import os
from stable_baselines3 import PPO
import gymnasium as gym
import gym_pcgrl  # Ensure that the PCGRL environments are registered
from PIL import Image
import numpy as np
from model import FullyConvPolicy  # Import your custom policy if needed
from utils import get_exp_name, max_exp_idx, load_model  # Utility functions

# Define game and representation
game = 'sokoban'
representation = 'narrow'  # Options: 'narrow', 'turtle', 'wide'
render_mode = 'rgb_array'  # Options: 'rgb_array', 'text_map'

# Create environment name
env_name = f"{game}-{representation}-v0"

# Create the environment using the same parameters as during training
env = gym.make(env_name, rep=representation, render_mode=render_mode)

# Load the trained model
exp_name = get_exp_name(game=game, representation=representation, experiment=None)
n = max_exp_idx(exp_name)
model_dir = f"runs/{game}_{0}_log_{representation}"

if os.path.exists(os.path.join(model_dir, 'best_model.zip')):
    print(f"Loading model from {model_dir}")
    model = PPO.load(os.path.join(model_dir, 'best_model'))
    print("Successfully loaded model")
else:
    raise FileNotFoundError(f"No model found in {model_dir}")

# For 'wide' representation, 'cropped_size' may not be relevant
# Verify observation space consistency
print(f"Environment observation space: {env.observation_space}")
print(f"Model's observation space: {model.observation_space}")

# Create a directory to save generated levels
os.makedirs(f'generated_levels_{representation}', exist_ok=True)
os.makedirs(f"generated_levels_{representation}/img", exist_ok=True)
os.makedirs(f"generated_levels_{representation}/txt", exist_ok=True)

# Generate levels
num_levels = 1000  # Number of levels to generate
for i in range(num_levels):
    obs, info = env.reset()
    print(f"\nGenerating Level {i}:")
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    
    # Save the generated level
    level = env.render()

    print(env.render_mode)

    # If image:
    if env.render_mode == 'rgb_array':
        img = Image.fromarray(level)
        img.save(f"generated_levels_{representation}/img/level_{i}.png")
    # If text:
    elif env.render_mode == 'text_map':
        with open(os.path.join(f"generated_levels_{representation}/txt", f"level_{i}.txt"), 'w') as f:
            f.write(level)

    # Optionally, print evaluation metrics
    print(f"Level {i} generated with final reward: {reward}")
    print(f"Info: {info}")
