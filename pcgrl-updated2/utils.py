import os
import re
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from gymnasium import Env
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym_pcgrl import wrappers


class RenderMonitor(Monitor):
    # Wrapper for the environment to save data and optionally render.
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)

        if isinstance(env, Env):
            self.single_env = env
        elif hasattr(env, 'envs'):
            self.single_env = env.envs[0]
        else:
            raise ValueError("Invalid environment passed to RenderMonitor")

        super().__init__(self.single_env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.single_env.render()
        return super().step(action)

def get_action(obs, model, deterministic=True):
    # Get an action from the model based on the observation.
    action, _ = model.predict(obs, deterministic=deterministic)
    return action

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    def _thunk():
        crop_size = kwargs.get('cropped_size', 10)
        render = kwargs.get('render', False)
        filtered_kwargs = {
            key: value for key, value in kwargs.items()
            if key not in ['resume', 'logging', 'representation', 'render', 'cropped_size']  # Remove unsupported arguments
        }

        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **filtered_kwargs)
        else:
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **filtered_kwargs)

        return env

    return _thunk


def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
    if n_cpu > 1:
        env_lst = [make_env(env_name, representation, rank=i, log_dir=log_dir, **kwargs) for i in range(n_cpu)]
        env = SubprocVecEnv(env_lst) 
        env = VecMonitor(env, log_dir)
    else:
        env = DummyVecEnv([make_env(env_name, representation, rank=0, log_dir=log_dir, **kwargs)])  # Create DummyVecEnv first
        env.envs[0] = RenderMonitor(env.envs[0], rank=0, log_dir=log_dir, **kwargs)
    return env

def get_exp_name(game, representation, experiment, **kwargs):
    # Generate a unique experiment name for logging and saving.
    exp_name = f'{game}_{representation}'
    if experiment:
        exp_name = f'{exp_name}_{experiment}'
    return exp_name

def max_exp_idx(exp_name):
    # Get the maximum experiment index for a given experiment name.
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob(f'{log_dir}*')
    if not log_files:
        return 0
    log_ns = [int(re.search(r'_(\d+)', f).group(1)) for f in log_files if re.search(r'_(\d+)', f)]
    return max(log_ns) if log_ns else 0

def load_model(log_dir):
    # Load the most recent model from the specified directory.
    model_path = None
    for filename in ['latest_model.zip', 'best_model.zip']:
        candidate = os.path.join(log_dir, filename)
        if os.path.exists(candidate):
            model_path = candidate
            break
    if not model_path:
        raise Exception('No models are saved in the specified directory.')
    model = PPO.load(model_path)
    return model
