import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from model import CustomActorCriticPolicy
from stable_baselines3.common.callbacks import CallbackList


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
    load_dir = f'runs/sokoban_wide_22_log'
    os.makedirs(log_dir, exist_ok=True)

    kwargs['render'] = render
    if representation == "narrow":
        kwargs['cropped_size'] = kwargs.get('cropped_size', 10)

    # Load or create a new model
    model = load_model(load_dir) if kwargs.get('resume', False) else None
    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)

    if not model:
        model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log=log_dir)  # Make sure tensorboard_log is set
    else:
        model.set_env(env)

    # Training callbacks
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir) if logging else None
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
    n_cpu = 10
    kwargs = {'resume': False}

    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)