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
from PIL import Image
import gymnasium as gym
import gym_pcgrl 

import pandas as pd


class CustomCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"[CustomCallback] Step {self.n_calls}: Checking performance for potential model save.")
            try:
                # Adjusted to find files with ".monitor.csv" extension
                monitor_files = [f for f in os.listdir(self.log_dir) if f.endswith('.monitor.csv')]
                print(f"[CustomCallback] Found monitor files: {monitor_files}")

                x, y = [], []
                for mf in monitor_files:
                    file_path = os.path.join(self.log_dir, mf)
                    if os.path.isfile(file_path):
                        # print(f"[CustomCallback] Processing file: {file_path}")
                        try:
                            data = pd.read_csv(file_path, skiprows=1)
                            x += data['t'].tolist()
                            y += data['r'].tolist()
                            # print(f"[CustomCallback] Loaded {len(data)} rows from {file_path}.")
                        except Exception as e:
                            print(f"[CustomCallback] Error reading {mf}: {e}")

                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    # print(f"[CustomCallback] Last 100 rewards: {y[-100:]}")
                    # print(f"[CustomCallback] {x[-1]} timesteps")
                    print(f"[CustomCallback] Best mean reward so far: {self.best_mean_reward:.2f}")
                    print(f"[CustomCallback] Last mean reward per episode: {mean_reward:.2f}")

                    # Save model if it's the best so far
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print("[CustomCallback] Saving new best model...")
                        self.model.save(os.path.join(self.log_dir, 'best_model'))
                    else:
                        print("[CustomCallback] Saving latest model...")
                        self.model.save(os.path.join(self.log_dir, 'latest_model'))
                else:
                    print("[CustomCallback] No reward data found yet. Skipping model save.")
            except Exception as e:
                print(f"[CustomCallback] Monitor results not found or error occurred: {e}. Skipping model save.")
        return True

# Custom callback for rendering
class RenderCallback(BaseCallback):
    def __init__(self, render_freq=10000, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq
        os.makedirs("generated_levels", exist_ok=True)
        os.makedirs("generated_levels/img", exist_ok=True)
        os.makedirs("generated_levels/txt", exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            # Access the first environment's original env
            env = self.training_env.envs[0].env

            level = env.render()  # Get the RGB array of the current level
            
            # Save the level as an image
            #img =  Image.fromarray(level)
            #img.save(os.path.join("generated_levels/img", f"level_{int(self.n_calls / self.render_freq)}.png")

            # print text map
            print(level)
            # with open(os.path.join("generated_levels/txt", f"level_{int(self.n_calls / self.render_freq)}.txt"), 'w') as f:
            #     f.write(level)

        return True


def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    global log_dir

    env_name = f"{game}-{representation}-v0"  # Correct environment name
    exp_name = get_exp_name(game=game, representation=representation, experiment=experiment)  # Pass 'experiment' via kwargs
    print(f"Experiment name: {exp_name}")

    resume = kwargs.get('resume', True)
    render_freq = kwargs.get('render_freq', 10)

    policy = FullyConvPolicySmallMap
    kwargs['cropped_size'] = 10

    n = max_exp_idx(exp_name)
    if not resume:
        n += 1
    log_dir = f"runs/{exp_name}_{n}_log_{representation}"
    if not resume:
        os.makedirs(log_dir, exist_ok=True)

    used_dir = log_dir if logging else None

    if render:
        n_cpu = 1

    # Pass 'rep' instead of 'representation' in env_kwargs
    env = make_vec_env(
        env_name,
        n_envs=n_cpu,
        monitor_dir=log_dir,
        env_kwargs={'rep': representation}
    )

    print(f"Observation space: {env.observation_space}")

    # Load or initialize the model
    if resume and os.path.exists(os.path.join(log_dir, 'best_model.zip')):
        print(f"Loading model from {log_dir}")
        model = PPO.load(os.path.join(log_dir, 'best_model'), env=env)
        print("Successfully loaded model")
    else:
        model = PPO(policy, env, verbose=1, tensorboard_log=log_dir)  # Align tensorboard_log with log_dir

    # Prepare callbacks
    callbacks = []
    if logging:
        callbacks.append(CustomCallback(log_dir=log_dir))
    
    if render:
        callbacks.append(RenderCallback(render_freq=render_freq))  # Adjust frequency as needed

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
    n_cpu = 20 
    kwargs = {
        'resume': True,
        'n_levels': 10,
        'render_freq': 10,
        'mode': 'text_mode'
    }

    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)

