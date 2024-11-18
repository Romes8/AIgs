"""
Helper functions for train and utility modules.
"""
import os
import re
import glob
import numpy as np
from gym_pcgrl import wrappers
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files and optionally render the GUI.
    """
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        super().__init__(env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()
        return super().step(action)

def get_action(obs, env, model, action_type=True):
    """
    Get the action from the model based on the action type.
    :param obs: (np.ndarray) Observation from the environment.
    :param env: (gym.Env) The environment.
    :param model: (stable_baselines3.PPO) The trained model.
    :param action_type: (bool) True for deterministic, False for stochastic.
    :return: (np.ndarray) The selected action.
    """
    if action_type:
        action, _ = model.predict(obs, deterministic=True)
    else:
        action, _ = model.predict(obs)
    return action

def make_env(env_name, rank=0, log_dir=None, **kwargs):
    """
    Return a function that initializes the environment when called.
    :param env_name: (str) Environment name.
    :param rank: (int) The rank of the environment.
    :param log_dir: (str) Directory for logs.
    :param kwargs: Additional parameters for the environment wrapper.
    :return: (function) Function that initializes the environment.
    """
    def _thunk():
        crop_size = kwargs.get('cropped_size', 10)
        env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        if log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk

def make_vec_envs(env_name, log_dir, n_cpu, **kwargs):
    """
    Prepare a vectorized environment using multiple processes or a single thread.
    :param env_name: (str) Environment name.
    :param log_dir: (str) Directory for logs.
    :param n_cpu: (int) Number of parallel environments.
    :param kwargs: Additional parameters for the environment.
    :return: (VecEnv) A vectorized environment.
    """
    if n_cpu > 1:
        env_lst = [make_env(env_name, i, log_dir, **kwargs) for i in range(n_cpu)]
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, 0, log_dir, **kwargs)])
    return env

def get_exp_name(experiment, **kwargs):
    """
    Generate a unique experiment name based on the parameters.
    :param experiment: (str) Experiment name.
    :param kwargs: Additional parameters for the experiment.
    :return: (str) Generated experiment name.
    """
    return f"sokoban_{experiment}" if experiment else "sokoban"

def max_exp_idx(exp_name):
    """
    Get the highest experiment index for a given experiment name.
    :param exp_name: (str) Experiment name.
    :return: (int) Maximum index for the experiment.
    """
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob(f"{log_dir}_*")
    if len(log_files) == 0:
        return 0
    log_ns = [int(re.search(r"_(\d+)", f).group(1)) for f in log_files if re.search(r"_(\d+)", f)]
    return max(log_ns)

def load_model(log_dir):
    """
    Load a saved PPO model from a given log directory.
    :param log_dir: (str) Directory containing the saved model.
    :return: (PPO) The loaded model.
    """
    model_path = os.path.join(log_dir, 'latest_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.zip')
    if not os.path.exists(model_path):
        files = [f for f in os.listdir(log_dir) if '.zip' in f]
        if len(files) > 0:
            model_path = os.path.join(log_dir, files[-1])
        else:
            raise FileNotFoundError("No saved models found in the directory.")
    return PPO.load(model_path)
