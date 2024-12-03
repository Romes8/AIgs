import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from model import CustomActorCriticPolicy
from stable_baselines3.common.callbacks import CallbackList
import pandas as pd

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
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Custom callback for saving the best model based on training reward.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model.zip')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Check progress every `check_freq` steps
        if self.n_calls % self.check_freq == 0:
            print('checking for save')
            rewards = self.training_env.get_attr('episode_rewards')
            mean_reward = np.mean([np.mean(r) for r in rewards if r])

            # if self.verbose > 0:
                # print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

            # Save the best model if reward improves
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print("Saving new best model")
                self.model.save(self.save_path)
        return True
    
def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    # Environment and experiment setup
    env_name = f'{game}-{representation}-v0'
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    log_dir = f'runs/{exp_name}_{max_exp_idx(exp_name) + 1}_log'
    os.makedirs(log_dir, exist_ok=True)

    kwargs['render'] = render
    if representation == "narrow":
        kwargs['cropped_size'] = kwargs.get('cropped_size', 10)

    # Load or create a new model
    model = load_model(log_dir) if kwargs.get('resume', False) else None
    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)

    if not model:
        model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log=log_dir)  # Make sure tensorboard_log is set
    else:
        model.set_env(env)

    # Training callbacks
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir) if logging else None
    # Use CallbackList with a list of callbacks
    callbacks = CallbackList([save_callback]) if save_callback else None

    # Start training
    model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callbacks, progress_bar=True)

if __name__ == '__main__':
    game = 'sokoban'
    representation = 'wide'
    experiment = None
    steps = 1e8
    render = False # will be overriten if n_cpu > 1
    logging = True
    n_cpu = 20
    kwargs = {'resume': False}

    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)