from gymnasium import spaces
from gym_pcgrl.envs.reps.representation import Representation
import numpy as np

class WideRepresentation(Representation):
    """
    Wide Representation for Sokoban.
    Encodes the `MultiDiscrete` action space into `Discrete` for compatibility with Stable-Baselines3.
    """

    def __init__(self):
        super().__init__()
        self._width = 0
        self._height = 0
        self._num_tiles = 0

    def reset(self, width, height, prob):
        """
        Resets the map and initializes parameters for width, height, and num_tiles.
        """
        super().reset(width, height, prob)
        self._width = width
        self._height = height
        self._num_tiles = len(prob.get_tile_types())

    def get_action_space(self, width, height, num_tiles):
        """
        Encodes `MultiDiscrete` action space as `Discrete` for Stable-Baselines3 compatibility.

        Action space:
        - Flattened `Discrete` space: width * height * num_tiles
        """
        self._width = width
        self._height = height
        self._num_tiles = num_tiles
        return spaces.Discrete(width * height * num_tiles)

    def get_observation_space(self, width, height, num_tiles):
        """
        Observation space for the Wide representation.
        Includes a 2D map of tile numbers as a `Dict`.
        """
        return spaces.Dict({
            "map": spaces.Box(
                low=0,
                high=num_tiles - 1,
                shape=(height, width),
                dtype=np.uint8
            )
        })

    def get_observation(self):
        """
        Returns the current observation as a dictionary with the key "map".
        """
        return {
            "map": self._map.copy()
        }

    def update(self, action):
        """
        Updates the map based on the given action.

        - Decodes a `Discrete` action into `MultiDiscrete` (x, y, tile_value).
        - Modifies the map at the specified location if the tile value changes.

        Parameters:
        - action (int): Flattened action.

        Returns:
        - change (int): 1 if the map was modified, 0 otherwise.
        - x (int): x-coordinate of the action.
        - y (int): y-coordinate of the action.
        """
        total_tiles = self._width * self._height
        tile_value = action % self._num_tiles
        pos = action // self._num_tiles
        y = pos // self._width
        x = pos % self._width

        change = int(self._map[y][x] != tile_value)
        self._map[y][x] = tile_value
        return change, x, y

    def render(self, lvl_image, tile_size, border_size):
        """
        Renders the map for visualization.

        Parameters:
        - lvl_image: Base image of the level.
        - tile_size (int): Size of each tile in pixels.
        - border_size (tuple): Size of the borders in the level.
        """
        # This method can be extended as needed for specific rendering requirements.
        return lvl_image
