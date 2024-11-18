# train.py

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import LoadMonitorResultsError
from model import FullyConvPolicySmallMap  # Import custom policies
from utils import get_exp_name, max_exp_idx, load_model  # Utility functions

import gymnasium as gym
import gym_pcgrl 

import pandas as pd

# Custom callback for saving models and tracking rewards
class CustomCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            try:
                print("1")
                # Find all monitor files in log_dir
                monitor_files = [f for f in os.listdir(self.log_dir) if f.startswith('monitor')]
                x, y = [], []
                for mf in monitor_files:
                    file_path = os.path.join(self.log_dir, mf)
                    if os.path.isfile(file_path):
                        try:
                            data = pd.read_csv(file_path, skiprows=1)
                            x += data['t'].tolist()
                            y += data['r'].tolist()
                        except Exception as e:
                            print(f"Error reading {mf}: {e}")
                print("2")
                if len(x) > 0:
                    print("3")
                    mean_reward = np.mean(y[-100:])
                    print(f"{x[-1]} timesteps")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    # Save model if it's the best so far
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print("Saving new best model")
                        self.model.save(os.path.join(self.log_dir, 'best_model'))
                    else:
                        print("Saving latest model")
                        self.model.save(os.path.join(self.log_dir, 'latest_model'))
            except Exception as e:
                print(f"Monitor results not found yet or error occurred: {e}. Skipping model save.")
        return True

# Custom callback for rendering
class RenderCallback(BaseCallback):
    def __init__(self, render_freq=10000, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                # Access the first environment's original env
                env = self.training_env.envs[0].env
                env.render()  # Ensure your environment supports 'human' mode
            except Exception as e:
                # print(f"Rendering failed: {e}")
                raise e
        return True

def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    global log_dir

    # Sokoban-specific parameters (game is always "sokoban" now)
    env_name = "sokoban-narrow-v0"  # Correct environment name
    exp_name = get_exp_name(game=game, representation=representation, experiment=experiment)  # Pass 'experiment' via kwargs
    resume = kwargs.get('resume', False)

    # Policy selection (use the custom policy class directly)
    policy = FullyConvPolicySmallMap  # Use the custom policy class directly

    # Set cropped size for Sokoban or any other game
    kwargs['cropped_size'] = 10

    # Manage experiment logging and model loading
    n = max_exp_idx(exp_name)
    if not resume:
        n += 1
    log_dir = f"runs/{exp_name}_{n}_log"
    if not resume:
        os.makedirs(log_dir, exist_ok=True)

    used_dir = log_dir if logging else None

    # If rendering is enabled, set n_cpu=1
    if render:
        n_cpu = 1

    # Create the environment with Monitor wrapper
    env = make_vec_env(
        env_name,
        n_envs=n_cpu,
        monitor_dir=log_dir,  # Direct Monitor logs to log_dir
    )

    # Verify observation_space
    print(f"Observation space: {env.observation_space}")

    # Load or initialize the model
    if resume and os.path.exists(os.path.join(log_dir, 'latest_model.zip')):
        model = PPO.load(os.path.join(log_dir, 'latest_model'), env=env)
    else:
        model = PPO(policy, env, verbose=1, tensorboard_log=log_dir)  # Align tensorboard_log with log_dir

    # Prepare callbacks
    callbacks = []
    if logging:
        callbacks.append(CustomCallback(log_dir=log_dir))
    if render:
        callbacks.append(RenderCallback(render_freq=1000))  # Adjust frequency as needed

    if callbacks:
        callback = CallbackList(callbacks)
    else:
        callback = None

    # Train the model
    model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback)

    # Save the final model
    model.save(os.path.join(log_dir, 'final_model'))

################################## MAIN ########################################
if __name__ == '__main__':
    game = 'sokoban'  # Hardcoded to "sokoban"
    representation = 'narrow'  # Representation is still an argument
    experiment = None
    steps = int(1e8)
    render = True
    logging = True
    n_cpu = 4  # Will be overridden to 1 if render=True
    kwargs = {'resume': False}

    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
