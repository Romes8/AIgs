# generator.py

import os
from stable_baselines3 import PPO
import gymnasium as gym
import gym_pcgrl  # Ensure that the PCGRL environments are registered
from PIL import Image
import numpy as np
from utils import get_exp_name, max_exp_idx  # Utility functions

# FlattenActionWrapper to convert MultiDiscrete action space to Discrete
class FlattenActionWrapper(gym.Wrapper):
    """
    A wrapper that flattens a MultiDiscrete action space into a Discrete action space.
    Handles both MultiDiscrete and Discrete action spaces.
    """
    def __init__(self, env):
        super(FlattenActionWrapper, self).__init__(env)
        self.original_action_space = env.action_space

        if isinstance(self.original_action_space, gym.spaces.MultiDiscrete):
            # Calculate the total number of discrete actions
            self.action_space_sizes = self.original_action_space.nvec
            self.total_actions = int(np.prod(self.action_space_sizes))
            self.action_space = gym.spaces.Discrete(self.total_actions)
            self._needs_flatten = True
            print(f"FlattenActionWrapper: Flattening MultiDiscrete action space {self.original_action_space} into Discrete({self.total_actions})")
        else:
            # No need to flatten
            self.action_space = self.original_action_space
            self._needs_flatten = False
            print(f"FlattenActionWrapper: Action space is Discrete, no flattening needed.")

    def action(self, action):
        if self._needs_flatten:
            return self.unflatten_action(action)
        else:
            return action

    def unflatten_action(self, index):
        action = []
        for size in reversed(self.action_space_sizes):
            action.append(index % size)
            index = index // size
        return np.array(list(reversed(action)), dtype=self.original_action_space.dtype)

    def step(self, action):
        if self._needs_flatten:
            action = self.unflatten_action(action)
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Define game and representation
game = 'sokoban'
representation = 'wide'  # Options: 'narrow', 'turtle', 'wide'
render_mode = 'rgb_array'  # Options: 'rgb_array', 'text_map'

# Create environment name
env_name = f"{game}-{representation}-v0"

# Create the environment using the same parameters as during training
env = gym.make(env_name, rep=representation, render_mode=render_mode)
env = FlattenActionWrapper(env)  # Apply the same wrapper as in training

# Load the trained model
exp_name = get_exp_name(game=game, representation=representation, experiment=None)
n = max_exp_idx(exp_name)
model_dir = f"runs/{exp_name}_{8}_log_{representation}"

if os.path.exists(os.path.join(model_dir, 'latest_model.zip')):
    print(f"Loading model from {model_dir}")
    model = PPO.load(os.path.join(model_dir, 'latest_model'), env=env)
    print("Successfully loaded model")
else:
    raise FileNotFoundError(f"No model found in {model_dir}")

# Verify observation space consistency
print(f"Environment observation space: {env.observation_space}")
print(f"Model's observation space: {model.observation_space}")

# Create directories to save generated levels
os.makedirs(f'generated_levels_{representation}', exist_ok=True)
os.makedirs(f"generated_levels_{representation}/img", exist_ok=True)
os.makedirs(f"generated_levels_{representation}/txt", exist_ok=True)

# Generate levels
num_levels = 100  # Number of levels to generate
for i in range(num_levels):
    obs, info = env.reset()
    print(f"\nGenerating Level {i}:")
    done = False
    truncated = False
    step = 0
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=False)
        print(f"Step {step}: Action: {action}")
        obs, reward, done, truncated, info = env.step(action)
        step += 1
    
    # Save the generated level
    level = env.render()

    # If image:
    if render_mode == 'rgb_array' and isinstance(level, np.ndarray):
        img = Image.fromarray(level)
        img.save(f"generated_levels_{representation}/img/level_{i}.png")
    # If text:
    elif render_mode == 'text_map' and isinstance(level, str):
        with open(os.path.join(f"generated_levels_{representation}/txt", f"level_{i}.txt"), 'w') as f:
            f.write(level)
    else:
        print(f"Unhandled render output type: {type(level)}")

    # Optionally, print evaluation metrics
    print(f"Level {i} generated with final reward: {reward}, steps: {step}")
    print(f"Info: {info}")
