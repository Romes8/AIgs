# train.py

import os
import numpy as np
import gymnasium as gym
import gym_pcgrl  # Ensure this import is correct and gym-pcgrl is installed
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from model import FullyConvPolicy
from utils import get_exp_name, max_exp_idx  # Ensure these utilities are correctly implemented
from PIL import Image
import pandas as pd

# FlattenActionWrapper to convert MultiDiscrete action space to Discrete
class FlattenActionWrapper(gym.ActionWrapper):
    """
    A wrapper that flattens a MultiDiscrete action space into a Discrete action space.
    """
    def __init__(self, env):
        super(FlattenActionWrapper, self).__init__(env)
        self.original_action_space = env.action_space
        # Check if action space is MultiDiscrete
        if isinstance(self.original_action_space, gym.spaces.MultiDiscrete):
            # Calculate the total number of discrete actions
            self.action_space_sizes = self.original_action_space.nvec
            self.total_actions = int(np.prod(self.action_space_sizes))
            self.action_space = gym.spaces.Discrete(self.total_actions)
            self._needs_flatten = True
            print(f"[FlattenActionWrapper] Flattening MultiDiscrete action space {self.original_action_space} to Discrete({self.total_actions})")
        else:
            self.action_space = self.original_action_space
            self._needs_flatten = False

    def action(self, action):
        """
        Convert the flattened action into the original MultiDiscrete action.
        """
        if self._needs_flatten:
            unflattened_action = self.unflatten_action(action)
            print(f"[FlattenActionWrapper] Flattened action {action} unflattened to {unflattened_action}")
            return unflattened_action
        else:
            return action

    def unflatten_action(self, index):
        action = []
        original_index = index  # Store for logging
        for size in reversed(self.action_space_sizes):
            action.append(index % size)
            index = index // size
        unflattened = np.array(list(reversed(action)), dtype=self.original_action_space.dtype)
        print(f"[FlattenActionWrapper] Unflattened action {original_index} to {unflattened}")
        return unflattened

    def step(self, action):
        if self._needs_flatten:
            # Log the received flattened action
            print(f"[FlattenActionWrapper] Received flattened action: {action}")
            action = self.unflatten_action(action)
            # Log the unflattened action
            print(f"[FlattenActionWrapper] Passing unflattened action to env: {action}")
        else:
            print(f"[FlattenActionWrapper] Action does not need flattening: {action}")
        # Since gymnasium environments return 5 values, we should expect 5
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Custom callback for logging steps per episode
class StepCounterCallback(BaseCallback):
    """
    Custom callback for logging the number of steps taken in each episode.
    """
    def __init__(self, verbose=0):
        super(StepCounterCallback, self).__init__(verbose)
        self.episode_step_counts = []
        self.current_steps = 0

    def _on_step(self) -> bool:
        self.current_steps += 1
        dones = self.locals.get('dones')
        if dones is not None:
            for done in dones:
                if done:
                    self.episode_step_counts.append(self.current_steps)
                    if self.verbose > 0:
                        print(f"Episode finished after {self.current_steps} steps.")
                    # Log to TensorBoard
                    self.logger.record('episode/steps', self.current_steps)
                    # Reset step count for the next episode
                    self.current_steps = 0
        return True

    def _on_training_end(self) -> None:
        if len(self.episode_step_counts) > 0:
            avg_steps = np.mean(self.episode_step_counts)
            print(f"\nAverage steps per episode: {avg_steps:.2f}")
            print(f"Total episodes: {len(self.episode_step_counts)}")
        else:
            print("\nNo episodes were completed during training.")

# Custom callback for saving best model
class CustomCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"[CustomCallback] Step {self.n_calls}: Checking performance for potential model save.")
            # Adjusted to find files with ".monitor.csv" extension
            monitor_files = [f for f in os.listdir(self.log_dir) if f.endswith('.monitor.csv')]
            print(f"[CustomCallback] Found monitor files: {monitor_files}")

            x, y = [], []
            for mf in monitor_files:
                file_path = os.path.join(self.log_dir, mf)
                if os.path.isfile(file_path):
                    try:
                        data = pd.read_csv(file_path, skiprows=1)
                        x += data['l'].tolist()  # 'l' is length of episode
                        y += data['r'].tolist()  # 'r' is reward
                    except Exception as e:
                        print(f"[CustomCallback] Error reading {mf}: {e}")

            if len(y) > 0:
                mean_reward = np.mean(y[-100:])

                # Save model if it's the best so far
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.log_dir, 'best_model'))
                else:
                    self.model.save(os.path.join(self.log_dir, 'latest_model'))
        return True

# Custom callback for rendering levels during training (optional)
class RenderCallback(BaseCallback):
    def __init__(self, render_freq=1, representation='turtle', verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.representation = representation
        os.makedirs(f"generated_levels_{self.representation}/img", exist_ok=True)
        os.makedirs(f"generated_levels_{self.representation}/txt", exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                # Access the first environment's original env
                env = self.training_env.envs[0].env

                level = env.render()  # Get the RGB array or string of the current level
                if isinstance(level, np.ndarray):
                    # Save the level as an image
                    try:
                        img = Image.fromarray(level)
                        img_path = os.path.join(f"generated_levels_{self.representation}/img", f"level_{int(self.n_calls / self.render_freq)}.png")
                        img.save(img_path)
                        print(f"[RenderCallback] Saved image: {img_path}")
                    except Exception as e:
                        print(f"[RenderCallback] Error saving image: {e}")
                elif isinstance(level, str):
                    # Save the text map
                    try:
                        txt_path = os.path.join(f"generated_levels_{self.representation}/txt", f"level_{int(self.n_calls / self.render_freq)}.txt")
                        with open(txt_path, 'w') as f:
                            f.write(level)
                        print(f"[RenderCallback] Saved text map: {txt_path}")
                    except Exception as e:
                        print(f"[RenderCallback] Error saving text map: {e}")
                else:
                    print(f"[RenderCallback] Unhandled level type: {type(level)}")
            except Exception as e:
                print(f"[RenderCallback] Error accessing environment for rendering: {e}")
        return True

def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    global log_dir

    env_name = f'{game}-{representation}-v0'
    exp_name = get_exp_name(game=game, representation=representation, experiment=experiment)
    print(f"Experiment name: {exp_name}")

    resume = kwargs.get('resume', False)
    render_freq = kwargs.get('render_freq', 10)

    features_dim = 256 if representation == 'wide' else 512

    # Create a single environment to get observation and action spaces
    base_env = gym.make(env_name, rep=representation)
    base_env = FlattenActionWrapper(base_env)  # Apply the wrapper here
    ob_space = base_env.observation_space
    ac_space = base_env.action_space

    print(f"Observation space: {ob_space}")
    print(f"Action space: {ac_space}")

    policy_kwargs = dict(
        features_extractor_kwargs=dict(
            features_dim=features_dim
        )
    )

    # Get the next experiment index
    n = max_exp_idx(exp_name)
    if not resume:
        n += 1
    log_dir = f"runs/{exp_name}_{n}_log_{representation}"
    if not resume:
        os.makedirs(log_dir, exist_ok=True)

    used_dir = log_dir if logging else None

    if render:
        n_cpu = 1

    # Define a function to create environments with the FlattenActionWrapper
    def make_env():
        env = gym.make(env_name, rep=representation)
        env = FlattenActionWrapper(env)
        return env

    # Create vectorized environments
    env = make_vec_env(
        make_env,
        n_envs=n_cpu,
        monitor_dir=log_dir,
    )

    model_path = os.path.join(log_dir, 'best_model.zip')
    if resume and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env)
        print("Successfully loaded model")
    else:
        model = PPO(
            FullyConvPolicy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            ent_coef=0.01,  # Adjusted to encourage exploration
        )

    print(f"Logging parameter count for policy: {FullyConvPolicy.__name__}")
    param_count = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count}")

    callbacks = []
    if logging:
        callbacks.append(CustomCallback(log_dir=log_dir))
        callbacks.append(StepCounterCallback())
    if render:
        callbacks.append(RenderCallback(render_freq=render_freq, representation=representation))

    callback = CallbackList(callbacks) if callbacks else None

    try:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback, progress_bar=True)
    except Exception as e:
        print(f"An error occurred during training: {e}")

    model.save(os.path.join(log_dir, 'final_model'))

################################## MAIN ########################################
if __name__ == '__main__':
    game = 'sokoban'
    representation = 'wide'
    experiment = None
    steps = int(1e8)
    render = False
    logging = True
    n_cpu = 50  # Adjust based on your hardware capabilities
    kwargs = {
        'resume': False,
    }
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
