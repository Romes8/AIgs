from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import gym
import gym_sokoban  # Ensure gym-sokoban is imported
from gym import spaces

class PcgrlEnv(gym.Env):
    """
    The PCGRL Gym Environment based on gym-sokoban's Sokoban-v0
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, prob="sokoban", rep="narrow"):
        super(PcgrlEnv, self).__init__()
        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._heatmap = np.zeros((self._prob._height, self._prob._width), dtype=np.uint8)

        self.seed()
        self.viewer = None

        # Define action and observation spaces
        self.action_space = self._rep.get_action_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )
        self.observation_space = self._rep.get_observation_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )
        self.observation_space.spaces['heatmap'] = spaces.Box(
            low=0, high=self._max_changes, dtype=np.uint8,
            shape=(self._prob._height, self._prob._width)
        )

    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._changes = 0
        self._iteration = 0
        self._rep.reset(
            self._prob._width,
            self._prob._height,
            get_int_prob(self._prob._prob, self._prob.get_tile_types())
        )
        self._rep_stats = self._prob.get_stats(
            get_string_map(self._rep._map, self._prob.get_tile_types())
        )
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros((self._prob._height, self._prob._width), dtype=np.uint8)

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        info = {}
        return observation, info

    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(
                int(percentage * self._prob._width * self._prob._height), 1
            )
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )
        self.observation_space = self._rep.get_observation_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )
        self.observation_space.spaces['heatmap'] = spaces.Box(
            low=0, high=self._max_changes, dtype=np.uint8,
            shape=(self._prob._height, self._prob._width)
        )

    def step(self, action):
        self._iteration += 1
        old_stats = self._rep_stats
        change, x, y = self._rep.update(action)

        if change > 0:
            self._changes += change
            self._heatmap[y, x] += 1
            self._rep_stats = self._prob.get_stats(
                get_string_map(self._rep._map, self._prob.get_tile_types())
            )

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()

        reward = self._prob.get_reward(self._rep_stats, old_stats)
        terminated = self._prob.get_episode_over(self._rep_stats, old_stats)
        truncated = (
            self._changes >= self._max_changes or
            self._iteration >= self._max_iterations
        )
        info = self._prob.get_debug_info(self._rep_stats, old_stats)
        info.update({
            "iterations": self._iteration,
            "changes": self._changes,
            "max_iterations": self._max_iterations,
            "max_changes": self._max_changes,
        })

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        tile_size = 16
        img = self._prob.render(
            get_string_map(self._rep._map, self._prob.get_tile_types())
        )
        img = self._rep.render(
            img, self._prob._tile_size, self._prob._border_size
        ).convert("RGB")

        if mode == 'rgb_array':
            return np.array(img)
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(np.array(img))
            return self.viewer.isopen()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
