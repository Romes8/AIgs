# gym_pcgrl/envs/pcgrl_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class PcgrlEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, prob="sokoban", rep="narrow", **kwargs):
        super(PcgrlEnv, self).__init__()

        if prob not in PROBLEMS:
            raise ValueError(f"Invalid problem name: {prob}. Must be one of {list(PROBLEMS.keys())}.")
        if rep not in REPRESENTATIONS:
            raise ValueError(f"Invalid representation name: {rep}. Must be one of {list(REPRESENTATIONS.keys())}.")

        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._heatmap = np.zeros((self._prob._height, self._prob._width), dtype=np.uint8)

        self.seed(kwargs.get('seed', None))
        self.viewer = None

        # Initialize observation space
        map_obs_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())

        heatmap_space = spaces.Box(
            low=0,
            high=self._max_changes,
            dtype=np.uint8,
            shape=(self._prob._height, self._prob._width),
        )

        # Combine map observation and heatmap into a single Box space by stacking along the channels
        if isinstance(map_obs_space, spaces.Box):
            combined_shape = (
                map_obs_space.shape[0],
                map_obs_space.shape[1],
                map_obs_space.shape[2] + 1,  # Add one channel for the heatmap
            )
            self.observation_space = spaces.Box(
                low=0.0,
                high=max(self.get_num_tiles(), self._max_changes),
                shape=(self._prob._height, self._prob._width, 2),  # Only 2 channels: map and heatmap
                dtype=np.float32
            )
        else:
            raise TypeError("Representation's observation_space must be a gymnasium.spaces.Box")

        self.action_space = self._rep.get_action_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )

    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._changes = 0
        self._iteration = 0
        self._rep.reset(
            width=self._prob._width,
            height=self._prob._height,
            prob=list(self._prob._prob.values())  # Ensure probabilities are correct
        )
        self._rep_stats = self._prob.get_stats(
            get_string_map(self._rep._map, self._prob.get_tile_types())
        )
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros((self._prob._height, self._prob._width), dtype=np.float32)

        # Get the observation map from the representation
        map_observation = self._rep.get_observation()

        # Ensure map_observation has 2D shape
        if map_observation.ndim == 2:
            map_observation = map_observation[..., np.newaxis]  # Add channel dimension

        # Stack heatmap as an additional channel
        combined_observation = np.concatenate(
            (map_observation, self._heatmap[..., np.newaxis]),  # Only 2 channels
            axis=2
        )

        return combined_observation, {}

    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)
        if 'max_iterations' in kwargs:
            self._max_iterations = kwargs.get('max_iterations', self._max_iterations)
        self._rep.adjust_param(**kwargs)
        self._prob.adjust_param(**kwargs)

    def step(self, action):
        self._iteration += 1

        # Save copy of old stats to calculate reward
        old_stats = self._rep_stats

        # Update the current state to the new state based on the action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[y][x] = min(self._heatmap[y][x] + 1, self._max_changes)

        # Get new stats
        new_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        reward = self._prob.get_reward(new_stats, old_stats)
        done = (
            self._prob.get_episode_over(new_stats, old_stats)
            or self._changes >= self._max_changes
            or self._iteration >= self._max_iterations
        )
        info = self._prob.get_debug_info(new_stats, old_stats)

        # Ensure map_observation has 3 dimensions (add channel if necessary)
        map_observation = self._rep.get_observation()
        if map_observation.ndim == 2:
            map_observation = map_observation[..., np.newaxis]

        # Stack heatmap as an additional channel
        combined_observation = np.concatenate(
            (map_observation, self._heatmap[..., np.newaxis]), axis=2
        )

        # Update stats
        self._rep_stats = new_stats

        return combined_observation, reward, done, False, info

    def render(self, mode='rgb_array'):
        tile_size = 16
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        
        if mode == 'rgb_array':
            return np.array(img)
        elif mode == 'human':
            if not hasattr(self, '_plt_fig'):
                # Initialize matplotlib figure and axes
                self._plt_fig, self._plt_ax = plt.subplots()
                self._plt_img = self._plt_ax.imshow(np.array(img))
                plt.ion()  # Turn on interactive mode
                plt.show()
            else:
                # Update the existing plot
                self._plt_img.set_data(np.array(img))
                self._plt_fig.canvas.draw()
                self._plt_fig.canvas.flush_events()
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not implemented.")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
