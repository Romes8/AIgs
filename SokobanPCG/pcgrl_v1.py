# first poor attempt

import numpy as np
from jumanji.environments import sokoban
import matplotlib.image as mpimg  
import matplotlib.pyplot as plt
from jax import jit
from PIL import Image
import random
import numpy as np
import random as py_random
from jax import random
import flax.linen as nn
import jax.numpy as jnp
import optax
import jax
from collections import deque, namedtuple
from tqdm import tqdm
import pickle
import os

from utils import OBJECT_TYPES, GRID_SIZE, assets, save_checkpoint, load_checkpoint

# save/load function
def save_checkpoint(file_path, params, target_params, opt_state):
    with open(file_path, 'wb') as f:
        pickle.dump({'params': params, 'target_params': target_params, 'opt_state': opt_state}, f)
    print(f"Checkpoint saved to {file_path}")

def load_checkpoint(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Checkpoint loaded from {file_path}")
        return checkpoint['params'], checkpoint['target_params'], checkpoint['opt_state']
    else:
        print(f"No checkpoint found at {file_path}. Starting from scratch.")
        return None, None, None
    
def map_level_to_image(level_classes):
    image_grid = np.zeros((level_classes.shape[0] * 32, level_classes.shape[1] * 32, 3), dtype=np.uint8)
    
    for i in range(level_classes.shape[0]):
        for j in range(level_classes.shape[1]):
            object_type = level_classes[i, j]
            asset_image = assets[list(OBJECT_TYPES.keys())[object_type]]
            
            if asset_image.dtype != np.uint8:
                asset_image = (asset_image * 255).astype(np.uint8)
            
            if asset_image.shape[0] != 32 or asset_image.shape[1] != 32:
                asset_image = np.array(Image.fromarray(asset_image).resize((32, 32)))

            image_grid[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32, :] = asset_image[:, :, :3]
    
    return image_grid


# each step corresponds to placing an object in one cell
# enviroment class
class SokobanEnv:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)  # empty grid
        self.current_position = (0, 0)  # start at top-left corner
        self.step_count = 0
        self.done = False
        # print(f"Environment reset. Grid size: {self.grid_size}")
        return self.grid

    def get_valid_actions(self):
        x, y = self.current_position
        valid_actions = np.ones(len(OBJECT_TYPES), dtype=bool)

        if self.grid[x, y] == OBJECT_TYPES['wall']:
            valid_actions[OBJECT_TYPES['wall']] = False
        if self.grid[x, y] == OBJECT_TYPES['box']:
            valid_actions[OBJECT_TYPES['box']] = False

        return valid_actions

    def step(self, action):
        x, y = self.current_position
        grid_size_x, grid_size_y = self.grid_size
        
        # print(f"Step {self.step_count + 1}: Current Position: {self.current_position}, Action: {action}")
        if action == OBJECT_TYPES['empty']:
            self.grid[x, y] = OBJECT_TYPES['empty']
        elif action == OBJECT_TYPES['wall']:
            self.grid[x, y] = OBJECT_TYPES['wall']
        elif action == OBJECT_TYPES['target']:
            self.grid[x, y] = OBJECT_TYPES['target']
        elif action == OBJECT_TYPES['agent']:
            self.grid[x, y] = OBJECT_TYPES['agent']
        elif action == OBJECT_TYPES['box']:
            self.grid[x, y] = OBJECT_TYPES['box']
        
        # print(f"Grid after placing object at {self.current_position}:")
        # print(self.grid)
        
        if y < grid_size_y - 1:
            self.current_position = (x, y + 1)
        elif x < grid_size_x - 1:
            self.current_position = (x + 1, 0)
        else:
            self.done = True  # grid is full
            reward = self.calculate_reward()  # calculate reward at the end
            return self.grid, reward, self.done

        self.step_count += 1
        return self.grid, 0, self.done

    def calculate_reward(self):
        reward = 0

        # reward for walls around edges
        for i in range(self.grid_size[0]):
            if self.grid[i, 0] == OBJECT_TYPES['wall']:
                reward += 1
            if self.grid[i, self.grid_size[1] - 1] == OBJECT_TYPES['wall']:
                reward += 1
        for j in range(self.grid_size[1]):
            if self.grid[0, j] == OBJECT_TYPES['wall']:
                reward += 1
            if self.grid[self.grid_size[0] - 1, j] == OBJECT_TYPES['wall']:
                reward += 1

        # reward for exactly one agent
        agent_count = np.sum(self.grid == OBJECT_TYPES['agent'])
        if agent_count == 1:
            reward += 10
        elif agent_count > 1:
            reward -= 5  # minus for more than one agent

        #matching number of boxes and targets
        box_count = np.sum(self.grid == OBJECT_TYPES['box'])
        target_count = np.sum(self.grid == OBJECT_TYPES['target'])

        if box_count == target_count:
            reward += 5
        else:
            reward -= 3

        return reward

    def display_level(self):
        img = map_level_to_image(self.grid)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Step: {self.step_count}")
        plt.show()

# model for tile placement
class DQNModel(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

def select_action(rng, q_values, epsilon, valid_actions):
    rng, subkey = random.split(rng)
    
    epsilon = jnp.asarray(epsilon)
    
    valid_q_values = jnp.where(valid_actions, q_values, -jnp.inf)
    
    # epsilon-greedy action selection
    if random.uniform(subkey, (), minval=0.0, maxval=1.0) < epsilon:
        # randomly pick an action
        valid_indices = jnp.nonzero(valid_actions)[0]
        random_action = py_random.choice(valid_indices)
        # print(f"Random action selected: {random_action}")
        return random_action
    else:
        # select the best valid action
        best_action = jnp.argmax(valid_q_values).item()
        # print(f"Best action selected: {best_action}")
        return best_action

@jit
def train_step(params, target_params, batch, opt_state):
    def loss_fn(params):
        q_values = model.apply(params, batch["obs"].reshape(-1, GRID_SIZE[0] * GRID_SIZE[1]))
        q_values = jnp.take_along_axis(q_values, batch["action"], axis=-1)
        q_values = jnp.squeeze(q_values, axis=-1)

        next_q_values = model.apply(target_params, batch["next_obs"].reshape(-1, GRID_SIZE[0] * GRID_SIZE[1]))
        target_q_values = batch["reward"] + (1.0 - batch["done"]) * gamma * jnp.max(next_q_values, axis=-1)

        return jnp.mean((q_values - target_q_values) ** 2)

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# initialize environment and model
learning_rate = 0.0003 
epsilon = 1.0  # initial exploration rate
epsilon_decay = 0.995  # decay rate
min_epsilon = 0.1 # exploration rate

batch_size = 32
gamma = 0.99
target_update_freq = 10

env = SokobanEnv(grid_size=GRID_SIZE)
action_dim = len(OBJECT_TYPES)
model = DQNModel(action_dim=action_dim)

key, rng = random.split(random.PRNGKey(0))
params = model.init(key, jnp.ones((1, GRID_SIZE[0] * GRID_SIZE[1])))
target_params = params

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# try to load a checkpoint, if exists
checkpoint_file = 'sokoban_dqn_checkpoint.pkl'
params, target_params, opt_state = load_checkpoint(checkpoint_file)
if params is None:
    print("Starting from scratch.")
    params = model.init(key, jnp.ones((1, GRID_SIZE[0] * GRID_SIZE[1])))
    target_params = params
    opt_state = optimizer.init(params)

entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
memory = deque(maxlen=10000)
generated_levels = []

# training Loop
episodes = 50
for episode in tqdm(range(episodes)):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        valid_actions = env.get_valid_actions()

        q_values = model.apply(params, obs.flatten()[None, :])
        
        # select action based on valid actions
        action = select_action(rng, q_values, epsilon, valid_actions)

        next_obs, reward, done = env.step(action)
        total_reward += reward

        # store the transition in memory
        memory.append(entry(obs, action, reward, next_obs, done))
        obs = next_obs

        if len(memory) > batch_size:
            batch = py_random.sample(memory, batch_size)
            batch = {
                "obs": jnp.array([entry.obs.flatten() for entry in batch]),
                "action": jnp.array([entry.action for entry in batch]).reshape(-1, 1),
                "reward": jnp.array([entry.reward for entry in batch]),
                "next_obs": jnp.array([entry.next_obs.flatten() for entry in batch]),
                "done": jnp.array([entry.done for entry in batch], dtype=jnp.float32),
            }
            params, opt_state = train_step(params, target_params, batch, opt_state)

    if episode % target_update_freq == 0:
        target_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

    # decay epsilon for exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print(f"Episode {episode+1}, Reward: {total_reward}")

    # save checkpoint every 50 episodes
    if (episode + 1) % 50 == 0:
        save_checkpoint(checkpoint_file, params, target_params, opt_state)

    # once the level is done, store it
    if done:
        generated_levels.append(env.grid.copy())  # store generated grid

#  display the last 50 levels
fig, axes = plt.subplots(5, 10, figsize=(20, 10)) 
axes = axes.flatten()
last_50_levels = generated_levels[-50:]
for i in range(len(last_50_levels)):
    img = map_level_to_image(last_50_levels[i])
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"Level {i+1}")

plt.tight_layout()
plt.show()