# train.py

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap  # Updated imports
from utils import get_exp_name, max_exp_idx, load_model  # Ensure these utilities are correctly implemented
from PIL import Image
import gymnasium as gym
import gym_pcgrl 

import pandas as pd

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
                        try:
                            data = pd.read_csv(file_path, skiprows=1)
                            x += data['t'].tolist()
                            y += data['r'].tolist()
                        except Exception as e:
                            print(f"[CustomCallback] Error reading {mf}: {e}")

                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
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

    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game=game, representation=representation, experiment=experiment)
    print(f"Experiment name: {exp_name}")

    resume = kwargs.get('resume', False)
    render_freq = kwargs.get('render_freq', 10)

    if representation == 'wide':
        features_dim = 256
    else:
        features_dim = 512

    single_env = gym.make(env_name, rep=representation)
    ob_space = single_env.observation_space
    ac_space = single_env.action_space

    if isinstance(ac_space, gym.spaces.Discrete):
        n_tools = int(ac_space.n / (ob_space.shape[0] * ob_space.shape[1]))
    elif isinstance(ac_space, gym.spaces.MultiDiscrete):
        n_tools = len(ac_space.nvec)  # For MultiDiscrete, use the length of nvec
    elif isinstance(ac_space, gym.spaces.Box):
        n_tools = int(np.prod(ac_space.shape) / (ob_space.shape[0] * ob_space.shape[1]))
    else:
        raise ValueError(f"Unsupported action space type: {type(ac_space)}")

    n_tools = max(1, n_tools)
    print(f"Calculated n_tools: {n_tools}")
    print(f"Observation space: {ob_space}")
    print(f"Action space: {ac_space}")

    if representation == 'wide':
        policy = FullyConvPolicyBigMap
    else:
        policy = FullyConvPolicySmallMap

    policy_kwargs = dict(
        features_extractor_kwargs=dict(
            n_tools=n_tools,
            features_dim=features_dim
        )
    )

    n = max_exp_idx(exp_name)
    if not resume:
        n += 1
    log_dir = f"runs/{exp_name}_{n}_log_{representation}"
    if not resume:
        os.makedirs(log_dir, exist_ok=True)

    used_dir = log_dir if logging else None

    if render:
        n_cpu = 1

    env = make_vec_env(
        env_name,
        n_envs=n_cpu,
        monitor_dir=log_dir,
        env_kwargs={'rep': representation}
    )

    model_path = os.path.join(log_dir, 'best_model.zip')
    if resume and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env)
        print("Successfully loaded model")
    else:
        model = PPO(
            policy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs
        )

    print(f"Logging parameter count for policy: {policy.__name__}")
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
    representation = 'turtle' 
    experiment = None
    steps = int(1e8)
    render = False
    logging = True
    n_cpu = 10
    kwargs = {
        'resume': False,
    }
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
