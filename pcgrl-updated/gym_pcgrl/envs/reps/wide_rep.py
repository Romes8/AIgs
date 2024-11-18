from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gymnasium import spaces
import numpy as np

"""
The wide representation where the agent can pick the tile position and tile value at each update.
"""
class WideRepresentation(Representation):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self):
        super().__init__()

    """
    Gets the action space used by the wide representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        MultiDiscrete: the action space used by that wide representation which
        consists of the x position, y position, and the tile value
    """
    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([width, height, num_tiles])

    """
    Get the observation space used by the wide representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Box: the observation space used by that representation. A 2D array of tile numbers
    """
    def get_observation_space(self, width, height, num_tiles):
        # Add a channel dimension (e.g., shape=(height, width, 1))
        return spaces.Box(
            low=0,
            high=num_tiles - 1,
            shape=(height, width, 1),
            dtype=np.uint8
        )

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. A 2D array of tile numbers
    """
    def get_observation(self):
        # Add a channel dimension to the map
        return self._map[..., np.newaxis].copy()

    """
    Update the wide representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        tuple: (change (bool), x (int), y (int))
               change is True if the action changes the map, otherwise False.
    """
    def update(self, action):
        x, y, tile_value = action
        # Check if the action modifies the map
        change = self._map[y][x] != tile_value
        if change:
            self._map[y][x] = tile_value
        return int(change), x, y
