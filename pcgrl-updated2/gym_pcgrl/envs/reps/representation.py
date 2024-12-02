from gymnasium import spaces
import numpy as np


class Representation:
    """
    Base class for all map representations in PCGRL environments.
    """
    def __init__(self):
        self._map = None
        self._random = np.random  # Random state for reproducibility

    def reset(self, width, height, prob):
        """
        Initializes the map with dimensions and tile probabilities.

        Parameters:
        - width (int): Width of the map.
        - height (int): Height of the map.
        - prob (list[float]): Probability distribution over tile types.
        """
        self._map = np.zeros((height, width), dtype=np.uint8)
        if prob is None or len(prob) == 0:
            raise ValueError("Probability distribution 'prob' must be provided and non-empty.")
        for y in range(height):
            for x in range(width):
                print('prob',prob)
                print('map', self._map)
                prob_values = np.array(list(prob.values()))
                self._map[y][x] = self._random.choice(len(prob), p=prob_values)             

    def seed(self, seed=None):
        """
        Sets the random seed for reproducibility.

        Parameters:
        - seed (int): Random seed value.
        """
        self._random = np.random.RandomState(seed)

    def adjust_param(self, **kwargs):
        """
        Adjusts representation-specific parameters.
        Subclasses can override this method.
        """
        pass

    def get_observation_space(self, width, height, num_tiles):
        """
        Returns the observation space for the map.

        Parameters:
        - width (int): Width of the map.
        - height (int): Height of the map.
        - num_tiles (int): Number of tile types.

        Returns:
        - gymnasium.spaces.Box: Observation space.
        """
        return spaces.Box(low=0, high=num_tiles - 1, shape=(height, width), dtype=np.uint8)

    def get_action_space(self, width, height, num_tiles):
        """
        Abstract method for defining the action space.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Action space must be defined in a subclass.")

    def get_observation(self):
        """
        Returns a copy of the current map as the observation.

        Returns:
        - np.ndarray: Current map.
        """
        return self._map.copy()

    def update(self, action):
        """
        Abstract method for updating the map with an action.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Update must be defined in a subclass.")

    def render(self, lvl_image, tile_size, border_size):
        """
        Abstract method for rendering the map.
        Must be implemented by subclasses.

        Parameters:
        - lvl_image: Base image for rendering.
        - tile_size (int): Size of each tile in pixels.
        - border_size (tuple[int, int]): Size of the border around the map.
        """
        raise NotImplementedError("Render must be defined in a subclass.")
