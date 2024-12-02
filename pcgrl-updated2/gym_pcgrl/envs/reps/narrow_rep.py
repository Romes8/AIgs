from gymnasium import spaces
from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
import numpy as np


class NarrowRepresentation(Representation):
    """
    Narrow Representation for Sokoban.
    This representation includes positional updates and map modifications.
    """

    def __init__(self):
        super().__init__()
        self._random_tile = True  # Whether to move randomly or sequentially
        self._x = 0  # Current x position
        self._y = 0  # Current y position

    def reset(self, width, height, prob):
        """
        Resets the map and initializes the agent's position.
        """
        super().reset(width, height, prob)
        self._x = np.random.randint(width)
        self._y = np.random.randint(height)

    def get_action_space(self, width, height, num_tiles):
        """
        Action space includes moving or modifying tiles.
        - 0: No change.
        - 1 to `num_tiles`: Change the tile at the current position.
        """
        return spaces.Discrete(num_tiles + 1)

    def get_observation_space(self, width, height, num_tiles):
        """
        Observation space includes:
        - `pos`: The agent's position.
        - `map`: The current map state.
        """
        return spaces.Dict({
            "pos": spaces.Box(
                low=np.array([0, 0]),
                high=np.array([width - 1, height - 1]),
                dtype=np.uint8
            ),
            "map": spaces.Box(
                low=0,
                high=num_tiles - 1,
                dtype=np.uint8,
                shape=(height, width)
            )
        })

    def get_observation(self):
        """
        Returns the current observation as a dictionary with:
        - `"pos"`: The agent's position.
        - `"map"`: The current map state.
        """
        return {
            "pos": np.array([self._x, self._y], dtype=np.uint8),
            "map": self._map.copy()
        }

    def adjust_param(self, **kwargs):
        """
        Adjusts the representation-specific parameters.
        """
        super().adjust_param(**kwargs)
        self._random_tile = kwargs.get('random_tile', self._random_tile)

    def update(self, action):
        """
        Updates the environment state based on the action.

        - `action == 0`: No change.
        - `action > 0`: Changes the tile at the current position.

        After an action:
        - The agent moves to a new position (randomly or sequentially).

        Returns:
        - `change` (int): 1 if the map was modified, 0 otherwise.
        - `x` (int): x-coordinate of the action.
        - `y` (int): y-coordinate of the action.
        """
        change = 0
        x, y = self._x, self._y

        if action > 0 and self._map[y][x] != action - 1:
            change = 1
            self._map[y][x] = action - 1

        # Move to the next position
        if self._random_tile:
            self._x = np.random.randint(self._map.shape[1])
            self._y = np.random.randint(self._map.shape[0])
        else:
            self._x = (self._x + 1) % self._map.shape[1]
            if self._x == 0:
                self._y = (self._y + 1) % self._map.shape[0]

        return change, x, y

    def render(self, lvl_image, tile_size, border_size):
        """
        Renders the agent's position on the map.
        """
        highlight = Image.new("RGBA", (tile_size, tile_size), (255, 0, 0, 255))
        lvl_image.paste(
            highlight,
            (
                (self._x + border_size[0]) * tile_size,
                (self._y + border_size[1]) * tile_size,
                (self._x + border_size[0] + 1) * tile_size,
                (self._y + border_size[1] + 1) * tile_size,
            ),
            highlight
        )
        return lvl_image