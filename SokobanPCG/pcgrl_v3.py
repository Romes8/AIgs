import numpy as np
import random as py_random
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import (OBJECT_TYPES, GRID_SIZE, assets, map_level_to_image, 
                   save_checkpoint, load_checkpoint, encode_level)

# Ensure reproducibility
np.random.seed(0)
py_random.seed(0)
jax.config.update("jax_enable_x64", True)

# PPO Policy Network
class PPOPolicy(nn.Module):
    action_space_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        # Policy head
        logits = nn.Dense(self.action_space_size)(x)

        # Value head
        value = nn.Dense(1)(x)

        return logits, value

class PPOAgent:
    def __init__(self, grid_size, learning_rate=3e-4, gamma=0.99,
                 lam=0.95, epsilon=0.2, epochs=10, batch_size=64, update_steps=2048):
        self.grid_size = grid_size
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.update_steps = update_steps

        # Create action space and calculate action_space_size
        self.action_space = self.create_action_space()
        self.action_space_size = len(self.action_space)
        self.action_to_index_map = {action: idx for idx, action in enumerate(self.action_space)}
        self.index_to_action_map = {idx: action for idx, action in enumerate(self.action_space)}

        # Initialize policy network with calculated action_space_size
        self.model = PPOPolicy(action_space_size=self.action_space_size)

        # Initialize parameters
        self.key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, np.prod(self.grid_size)))
        self.params = self.model.init(self.key, dummy_input)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

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

    def get_action(self, params, state, key):
        logits, value = self.model.apply(params, state)
        action_dist = jax.nn.softmax(logits)
        action_index = jax.random.categorical(key, logits)
        action_index = int(action_index.item())
        log_prob = jnp.log(action_dist[0, action_index] + 1e-8)
        return action_index, log_prob, value[0, 0]

    def select_action(self, state):
        state_jnp = jnp.array(state[None, :])
        self.key, subkey = jax.random.split(self.key)
        action_index, log_prob, value = self.get_action(self.params, state_jnp, subkey)
        action = self.index_to_action(action_index)
        return action, float(log_prob), float(value)

    def store_transition(self, state, action_index, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action_index)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self):
        states = jnp.array(self.states)
        actions = jnp.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        log_probs = jnp.array(self.log_probs)
        dones = np.array(self.dones)

        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = jnp.array(advantages)
        returns = jnp.array(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(self.states)
        num_batches = max(1, num_samples // self.batch_size)

        for epoch in range(self.epochs):
            permutation = np.random.permutation(num_samples)
            for i in range(num_batches):
                batch_indices = permutation[i * self.batch_size:(i + 1) * self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                loss_value, grads = jax.value_and_grad(self.loss_fn)(
                    self.params,
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_returns,
                    batch_advantages
                )
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
                self.params = optax.apply_updates(self.params, updates)

        # Clear memory buffers after update
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def loss_fn(self, params, states, actions, old_log_probs, returns, advantages):
        logits, values = self.model.apply(params, states)
        values = values.squeeze()

        # Compute the action distribution
        action_dists = jax.nn.softmax(logits)
        action_log_probs = jnp.log(action_dists[jnp.arange(len(actions)), actions] + 1e-8)

        # Compute the ratio (pi_theta / pi_theta_old)
        ratios = jnp.exp(action_log_probs - old_log_probs)

        # Compute policy loss
        surrogate1 = ratios * advantages
        surrogate2 = jnp.clip(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

        # Compute value function loss
        value_loss = jnp.mean((returns - values) ** 2)

        # Entropy bonus
        entropy = -jnp.mean(jnp.sum(action_dists * jnp.log(action_dists + 1e-8), axis=1))
        entropy_coefficient = 0.01

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - entropy_coefficient * entropy

        return total_loss

# Problem Module with Updated Reward Function
class ProblemModule:
    def __init__(self):
        self.goal_agent_count = 1

    def calculate_reward(self, grid):
        reward = 0
        agent_count = np.sum(grid == OBJECT_TYPES['agent'])
        box_count = np.sum(grid == OBJECT_TYPES['box'])
        target_count = np.sum(grid == OBJECT_TYPES['target'])

        # Reward for correct agent count
        if agent_count == self.goal_agent_count:
            reward += 1
        else:
            reward -= abs(agent_count - self.goal_agent_count) * 0.1

        # Reward for balanced boxes and targets
        balance = box_count - target_count
        if balance == 0 and box_count > 0:
            reward += 1
        else:
            reward -= abs(balance) * 0.1

        return reward

    def is_balanced(self, grid):
        boxes = np.sum(grid == OBJECT_TYPES['box'])
        targets = np.sum(grid == OBJECT_TYPES['target'])
        return boxes == targets and boxes > 0

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
        # self.change_limit = int(np.prod(grid_size) * change_percentage)
        self.change_limit =100 
        print(f"Change limit: {self.change_limit}")
        self.problem_module = ProblemModule()
        self.reset()

    def reset(self):
        # Initialize grid with walls on edges and random tiles inside
        self.grid = np.random.choice(
            [OBJECT_TYPES['empty'], OBJECT_TYPES['box'], OBJECT_TYPES['target'], OBJECT_TYPES['agent']],
            size=self.grid_size
        )
        # Set walls on edges
        self.grid[0, :] = OBJECT_TYPES['wall']
        self.grid[-1, :] = OBJECT_TYPES['wall']
        self.grid[:, 0] = OBJECT_TYPES['wall']
        self.grid[:, -1] = OBJECT_TYPES['wall']

        self.change_count = 0
        self.done = False
        return self.grid

    def step(self, action):
        if self.change_count >= self.change_limit:
            self.done = True
            reward = self.problem_module.calculate_reward(self.grid)
            return self.grid, reward, self.done

        x, y, tile_type = action

        # Check for invalid action: placing same tile or modifying edges
        if self.grid[x, y] == tile_type or self.is_edge_position(x, y):
            reward = -0.1  # Penalize invalid action slightly
            self.change_count += 1
            return self.grid, reward, self.done

        self.grid[x, y] = tile_type
        self.change_count += 1
        reward = self.problem_module.calculate_reward(self.grid)

        if self.problem_module.is_goal_met(self.grid):
            self.done = True
            reward += 10  # Final reward if goal achieved

        # Check if change limit reached after the action
        if self.change_count >= self.change_limit:
            self.done = True

        return self.grid, reward, self.done

    def is_edge_position(self, x, y):
        max_x, max_y = self.grid_size
        return x == 0 or y == 0 or x == max_x - 1 or y == max_y - 1

# Initialize environment and agent
representation_module = RepresentationModule(grid_size=GRID_SIZE)
env = SokobanEnv(grid_size=GRID_SIZE, change_percentage=0.3)
agent = PPOAgent(grid_size=GRID_SIZE)

episodes = 3000
generated_levels = []
total_steps = 0

# Training loop with PPO agent
for episode in tqdm(range(episodes)):
    obs = env.reset()
    done = False
    total_reward = 0
    steps_in_episode = 0

    while not done:
        state = representation_module.get_state(env.grid)
        action, log_prob, value = agent.select_action(state)
        action_index = agent.action_to_index(action)

        next_obs, reward, done = env.step(action)
        next_state = representation_module.get_state(next_obs)

        agent.store_transition(state, action_index, reward, value, log_prob, done)

        total_reward += reward
        steps_in_episode += 1
        total_steps += 1

        # Update the agent after collecting enough steps
        if total_steps >= agent.update_steps:
            agent.update()
            total_steps = 0

    generated_levels.append(env.grid.copy())
    print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}, Steps: {steps_in_episode}")

# Visualization code (optional)
# Ensure you have a function to convert levels to images, or comment this out
fig, axes = plt.subplots(5, 10, figsize=(20, 10))
axes = axes.flatten()
for i in range(min(len(generated_levels), 50)):
    img = map_level_to_image(generated_levels[i])
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"Level {i+1}")

plt.tight_layout()
plt.show()
