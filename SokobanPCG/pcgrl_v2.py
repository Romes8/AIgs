# DQN attempt

import numpy as np
import random as py_random
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax import jit

from utils import (OBJECT_TYPES, GRID_SIZE, assets, map_level_to_image, 
                   save_checkpoint, load_checkpoint, encode_level, random_populate_grid)

# DQN Network Definition
class DQN(nn.Module):
    action_space_size: int  # Add action_space_size as a field

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_space_size)(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idxs]

    def size(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, grid_size, action_space_size, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 learning_rate=0.001, gamma=0.99, batch_size=32,
                 replay_buffer_capacity=10000, target_update_interval=10):
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.action_space_size = action_space_size
        self.model = DQN(action_space_size=self.action_space_size)
        self.target_model = DQN(action_space_size=self.action_space_size)

        # Initialize parameters
        self.key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, np.prod(self.grid_size)))
        self.params = self.model.init(self.key, dummy_input)
        self.target_params = self.params
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # Target network update frequency
        self.target_update_interval = target_update_interval
        self.training_steps = 0

        # Create action space and mappings
        self.action_space = self.create_action_space()
        self.action_to_index_map = {action: idx for idx, action in enumerate(self.action_space)}
        self.index_to_action_map = {idx: action for idx, action in enumerate(self.action_space)}

    def create_action_space(self):
        action_space = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for tile_type in OBJECT_TYPES.values():
                    action_space.append((x, y, tile_type))
        return action_space

    def action_to_index(self, action):
        return self.action_to_index_map[action]

    def index_to_action(self, index):
        return self.index_to_action_map[index]

    @jax.jit
    def get_best_action(self, params, state):
        q_values = self.model.apply(params, state)
        return jnp.argmax(q_values)

    def select_action(self, state, valid_actions):
        if py_random.random() < self.epsilon:
            return py_random.choice(valid_actions)
        else:
            state_jnp = jnp.array(state[None, :])  # Add batch dimension
            best_action_index = int(self.get_best_action(self.params, state_jnp))
            action = self.index_to_action(best_action_index)
            if action in valid_actions:
                return action
            else:
                # Get Q-values for valid actions
                q_values = self.model.apply(self.params, state_jnp)
                valid_indices = [self.action_to_index(a) for a in valid_actions]
                q_values_valid = q_values[0, valid_indices]
                best_valid_action_index = valid_indices[jnp.argmax(q_values_valid)]
                return self.index_to_action(best_valid_action_index)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = jnp.array(states)
        next_states = jnp.array(next_states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        dones = jnp.array(dones)

        @jax.jit
        def loss_fn(params, target_params, states, actions, rewards, next_states, dones):
            q_values = self.model.apply(params, states)
            next_q_values = self.target_model.apply(target_params, next_states)
            target_q_values = rewards + self.gamma * (1 - dones) * jnp.max(next_q_values, axis=-1)
            td_errors = q_values[jnp.arange(len(actions)), actions] - target_q_values
            loss = jnp.mean(td_errors ** 2)
            return loss

        loss_value, grads = jax.value_and_grad(loss_fn)(
            self.params, self.target_params, states, actions, rewards, next_states, dones)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.update_target_network()

        return loss_value

# Problem Module with Improved Reward Function
class ProblemModule:
    def __init__(self):
        self.goal_agent_count = 1

    def calculate_reward(self, grid, last_action_tile):
        reward = 0
        agent_count = np.sum(grid == OBJECT_TYPES['agent'])
        box_count = np.sum(grid == OBJECT_TYPES['box'])
        target_count = np.sum(grid == OBJECT_TYPES['target'])

        # Reward for correct agent count
        if agent_count == self.goal_agent_count:
            reward += 5
        else:
            reward -= 1

        # Reward for balanced boxes and targets
        if box_count == target_count and box_count > 0:
            reward += 5
        else:
            reward -= 1

        # Penalize adding a box if boxes exceed targets
        if last_action_tile == OBJECT_TYPES['box'] and box_count > target_count:
            reward -= 2

        # Penalize adding a target if targets exceed boxes
        if last_action_tile == OBJECT_TYPES['target'] and target_count > box_count:
            reward -= 2

        # Penalize if agent count exceeds goal
        if agent_count > self.goal_agent_count and last_action_tile == OBJECT_TYPES['agent']:
            reward -= 2

        return reward

    def is_balanced(self, grid):
        return np.sum(grid == OBJECT_TYPES['box']) == np.sum(grid == OBJECT_TYPES['target']) and np.sum(grid == OBJECT_TYPES['box']) > 0

    def is_goal_met(self, grid):
        return self.is_balanced(grid) and np.sum(grid == OBJECT_TYPES['agent']) == self.goal_agent_count


# Representation Module
class RepresentationModule:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def get_state(self, grid):
        return grid.flatten()

# Sokoban Environment
class SokobanEnv:
    def __init__(self, grid_size=GRID_SIZE, change_percentage=0.3):
        self.grid_size = grid_size
        self.change_limit = int(np.prod(grid_size) * change_percentage)
        print(f"Change limit: {self.change_limit}")
        self.problem_module = ProblemModule()
        self.reset()

    def reset(self):
        self.grid = random_populate_grid(self)
        self.change_count = 0
        self.done = False
        return self.grid

    def step(self, action):
        if self.change_count >= self.change_limit:
            self.done = True
            return self.grid, 0, self.done

        x, y, tile_type = action
        last_action_tile = tile_type

        # Check for invalid action: placing same tile
        if self.grid[x, y] == tile_type:
            reward = -2  # Penalize invalid action
            self.done = False
            return self.grid, reward, self.done

        self.grid[x, y] = tile_type
        self.change_count += 1
        reward = self.problem_module.calculate_reward(self.grid, last_action_tile)

        if self.problem_module.is_goal_met(self.grid):
            self.done = True
            reward += 20  # Final reward if goal achieved

        return self.grid, reward, self.done

    def get_valid_actions(self):
        valid_actions = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for tile_type in OBJECT_TYPES.values():
                    if self.grid[x, y] != tile_type:
                        valid_actions.append((x, y, tile_type))
        return valid_actions

# Initialize environment and agent
representation_module = RepresentationModule(grid_size=GRID_SIZE)
env = SokobanEnv(grid_size=GRID_SIZE, change_percentage=0.3)
agent = DQNAgent(grid_size=GRID_SIZE, action_space_size=len(OBJECT_TYPES)*GRID_SIZE[0]*GRID_SIZE[1])
episodes = 4000
generated_levels = []

# Training loop with DQN agent
for episode in tqdm(range(episodes)):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        state = representation_module.get_state(env.grid)
        valid_actions = env.get_valid_actions()

        action = agent.select_action(state, valid_actions)
        action_index = agent.action_to_index(action)

        next_obs, reward, done = env.step(action)
        next_state = representation_module.get_state(next_obs)

        agent.store_transition(state, action_index, reward, next_state, done)
        agent.train()

        total_reward += reward
        # Remove agent.update_epsilon() from here

    agent.update_epsilon()  # Update epsilon after the episode

    generated_levels.append(env.grid.copy())
    print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")
# Visualize the generated levels
fig, axes = plt.subplots(5, 10, figsize=(20, 10)) 
axes = axes.flatten()
for i in range(len(generated_levels)):
    img = map_level_to_image(generated_levels[i])
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"Level {i+1}")

plt.tight_layout()
plt.show()
