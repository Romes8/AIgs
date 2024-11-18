# test_env.py

import gymnasium as gym
import gym_pcgrl 

print("Registered environments:")
for env_spec in gym.envs.registry.values():
    print(env_spec.id)

# Test creating the environment
try:
    env_test = gym.make("sokoban-narrow-v0")
    print("Environment sokoban-narrow-v0 created successfully.")
    
    # Print observation space
    print(f"Observation space: {env_test.observation_space}")
    
    # Reset the environment
    obs, info = env_test.reset()
    print(f"Initial observation shape: {obs.shape}")
except gym.error.Error as e:
    print(f"Failed to create environment: {e}")
