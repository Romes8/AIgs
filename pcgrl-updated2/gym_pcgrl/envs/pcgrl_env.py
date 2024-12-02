import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import matplotlib.pyplot as plt

class PcgrlEnv(gym.Env):
    """
    Procedural Content Generation Reinforcement Learning (PCGRL) Environment.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, prob="sokoban", rep="narrow"):
        """
        Initializes the PCGRL environment with a specific problem and representation.

        Args:
            prob (str): The problem name, defined in PROBLEMS.
            rep (str): The representation name, defined in REPRESENTATIONS.
        """
        # Initialize problem and representation
        self._prob = PROBLEMS[prob]()
        print('REPS', REPRESENTATIONS)
        print('REP', rep)
        self._rep = REPRESENTATIONS[rep]()

        # Initialize environment parameters
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._heatmap = np.zeros((self._prob._height, self._prob._width), dtype=np.uint8)

        # Define action and observation spaces
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())

        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(
            low=0,
            high=self._max_changes,
            shape=(self._prob._height, self._prob._width),
            dtype=np.uint8
        )

        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        self._changes = 0
        self._iteration = 0
        self._rep.reset(self._prob._width, self._prob._height, get_int_prob(self._prob._prob, self._prob.get_tile_types()))
        self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros((self._prob._height, self._prob._width), dtype=np.uint8)

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()

        # Return observation and info
        info = {}  # Add any relevant info to this dictionary
        return observation, info

    def get_border_tile(self):
        """
        Returns the border tile for padding purposes.
        """
        return self._prob.get_tile_types().index(self._prob._border_tile)

    def get_num_tiles(self):
        """
        Returns the number of tile types in the problem.
        """
        return len(self._prob.get_tile_types())

    def adjust_param(self, **kwargs):
        """
        Adjusts parameters for the problem and representation.
        """
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage', 0)))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height

        # Adjust parameters in problem and representation
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)

        # Update action and observation spaces
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(
            low=0,
            high=self._max_changes,
            shape=(self._prob._height, self._prob._width),
            dtype=np.uint8
        )

    def step(self, action):
        """
        Advances the environment state based on the action.

        Args:
            action: The action to take.

        Returns:
            observation, reward, terminated, truncated, info: Tuple of environment state.
        """
        self._iteration += 1
        old_stats = self._rep_stats
        change, x, y = self._rep.update(action)
        # print(f"Agent moved to: ({x}, {y})")
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        
        # Terminate if max changes or iterations are exceeded
        terminated = self._prob.get_episode_over(self._rep_stats, old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        
        # Truncated flag can be used based on specific conditions
        truncated = False  # You may set this flag if there's a specific truncation condition
        
        info = self._prob.get_debug_info(self._rep_stats, old_stats)
        info.update({"iterations": self._iteration, "changes": self._changes})
        
        return observation, reward, terminated, truncated, info
    
    #  dont remove this
    # def render(self, mode='human'):
    #     print('Rendering', mode)
        
    #     # Generate image from the environment state
    #     img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
    #     img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        
    #     if mode == 'rgb_array':
    #         return np.array(img)
    #     elif mode == 'human':
    #         # Initialize matplotlib figure and axes if not already done
    #         if not hasattr(self, '_plt_fig'):
    #             self._plt_fig, self._plt_ax = plt.subplots()
    #             self._plt_img = self._plt_ax.imshow(np.array(img))
    #             self._plt_ax.axis('off')  # Hide the axes for cleaner output
    #             plt.ion()  # Turn on interactive mode to update the plot in real-time
    #             plt.show()
    #         else:
    #             # Update the existing plot
    #             self._plt_img.set_data(np.array(img))
    #             self._plt_fig.canvas.draw()
    #             self._plt_fig.canvas.flush_events()
    #         return True  # Return True to indicate successful rendering
    
    def render(self, mode='human'):
        print('Rendering', mode)
        
        # Generate image from the environment state
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        
        if mode == 'rgb_array':
            return np.array(img)
        elif mode == 'human':
            # Initialize matplotlib figure and axes if not already done
            if not hasattr(self, '_plt_fig') or not plt.fignum_exists(self._plt_fig.number):
                self._plt_fig, self._plt_ax = plt.subplots()
                self._plt_img = self._plt_ax.imshow(np.array(img))
                self._plt_ax.axis('off')  # Hide the axes for cleaner output
                plt.show()  # This will block until the plot is closed by the user
            else:
                # Update the existing plot
                self._plt_img.set_data(np.array(img))
                self._plt_fig.canvas.draw()
                self._plt_fig.canvas.flush_events()

            return True  # Return True to indicate successful rendering
        
    def close(self):
        """
        Closes the environment and releases resources.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

 