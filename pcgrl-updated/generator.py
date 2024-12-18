import os
from stable_baselines3 import PPO
from PIL import Image
import numpy as np
from utils import get_exp_name, max_exp_idx
from gym_pcgrl import wrappers
import time


game = 'sokoban'
representation = 'narrow'  # 'narrow', 'turtle', 'wide'

env_name = f"{game}-{representation}-v0"
exp_name = get_exp_name(game=game, representation=representation, experiment=None)
n = max_exp_idx(exp_name)

# model used for generation
model_dir = f"final_runs/{exp_name}_log"
# model_dir = "runs/sokoban_wide_24_log"

if not os.path.exists(model_dir):
    raise FileNotFoundError(f"The log directory {model_dir} does not exist. Ensure that the model has been trained and saved correctly.")

if representation == 'wide':
    env = wrappers.ActionMapImagePCGRLWrapper(env_name)
elif representation == 'narrow':
    env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size=10)
elif representation == 'turtle':
    env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size=10)
else:
    raise ValueError(f"Unsupported representation: {representation}")

# Load the trained model
model_path = os.path.join(model_dir, 'best_model.zip')
if os.path.exists(model_path):
    print(f"Loading model from {model_dir}")
    model = PPO.load(model_path, env=env)
    print("Successfully loaded model")
else:
    raise FileNotFoundError(f"No model found in {model_dir}")

# Create directories to save generated levels
img_dir = f'generated_levels_{representation}/img'
txt_dir = f'generated_levels_{representation}/txt'
os.makedirs(img_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)

pcgrl_env = env.unwrapped

# Generate levels
num_levels = 100 
solvable_count = 0
for i in range(num_levels):
    # Reset environment
    reset_output = env.reset()
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        obs, info = reset_output
    else:
        obs = reset_output[0]
        info = {}

    print(f"\nGenerating Level {i}:")
    done = False
    truncated = False
    step = 0
    total_reward = 0
    while not (done or truncated or step >= 50):
        # time.sleep(0.1)
        action, _states = model.predict(obs, deterministic=False)

        step_output = env.step(action)
        if isinstance(step_output, tuple):
            obs, reward, done, truncated, info = step_output
        else:
            obs, reward, done, info = step_output
            truncated = False
        total_reward += reward
        level = pcgrl_env.render(mode='human') 
        step += 1
    level = pcgrl_env.render(mode='rgb_array') 
    print ('Solvable:', info['solvable'])
    if info['solvable']:
        solvable_count += 1


    try:
        if level.dtype != np.uint8:
            level = (255 * (level - level.min()) / (level.max() - level.min())).astype(np.uint8)

        # Save the level as an image
        img_path = os.path.join(img_dir, f"level_{i}.png")
        Image.fromarray(level).save(img_path)

    except Exception as e:
        print(f"Failed to save level {i}: {e}")
    print(f"Saved level {i} as image: , total reward: {total_reward}")


print('Solvable levels:'+str(solvable_count)+'/'+str(num_levels)) 