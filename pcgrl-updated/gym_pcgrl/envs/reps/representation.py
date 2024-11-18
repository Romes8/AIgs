from gymnasium import spaces
import numpy as np

class Representation:
    def __init__(self):
        self._map = None
        self._random = np.random

    def reset(self, width, height, prob):
        self._map = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                self._map[y][x] = np.random.choice(len(prob), p=prob)

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)

    def adjust_param(self, **kwargs):
        pass

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Box(low=0, high=num_tiles - 1, shape=(height, width), dtype=np.uint8)

    def get_action_space(self, width, height, num_tiles):
        raise NotImplementedError("Action space must be defined in a subclass.")

    def get_observation(self):
        return self._map.copy()

    def update(self, action):
        raise NotImplementedError("Update must be defined in a subclass.")

    def render(self, lvl_image, tile_size, border_size):
        raise NotImplementedError("Render must be defined in a subclass.")
