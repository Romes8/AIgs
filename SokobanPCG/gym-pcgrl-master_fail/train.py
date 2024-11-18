import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
import gym_sokoban  # Ensure gym-sokoban is installed and imported
from gym_pcgrl.wrappers import ActionMapImagePCGRLWrapper  # Correct import from gym_pcgrl
from stable_baselines3.common.results_plotter import ts2xy, load_results

# Initialize global variables
n_steps = 0
log_dir = './logs/sokoban/'
best_mean_reward = -np.inf  # Initialize outside the loop

class CustomCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(save_freq=save_freq, save_path=save_path, verbose=verbose)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        global n_steps
        n_steps += 1
        if n_steps % 10 == 0:
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 100:
                mean_reward = np.mean(y[-100:])
                print(f"{x[-1]} timesteps")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # Save the best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print("Saving new best model")
                    self.model.save(os.path.join(log_dir, 'best_model.zip'))
                else:
                    print("Saving latest model")
                    self.model.save(os.path.join(log_dir, 'latest_model.zip'))
            else:
                print(f'{len(x)} monitor entries')
        return True

def make_env(rank):
    def _init():
        # Use the ActionMapImagePCGRLWrapper from gym_pcgrl.wrappers
        env = ActionMapImagePCGRLWrapper('Sokoban-v0')
        env = Monitor(env, log_dir)
        return env
    return _init

def main(steps, n_cpu, logging):
    os.makedirs(log_dir, exist_ok=True)

    # Create multiple environments
    env = SubprocVecEnv([make_env(i) for i in range(n_cpu)])

    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)

    # Initialize PPO with CnnPolicy suitable for image observations
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./runs")

    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name='sokoban_ppo')
    else:
        model.learn(
            total_timesteps=int(steps),
            tb_log_name='sokoban_ppo',
            callback=CustomCallback(save_freq=1000, save_path=log_dir)
        )

if __name__ == '__main__':
    steps = int(1e7)  # 10 million steps
    logging = True
    n_cpu = 4  # Adjust based on your system's capabilities
    main(steps, n_cpu, logging)
