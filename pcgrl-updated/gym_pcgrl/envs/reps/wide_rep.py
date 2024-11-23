from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gymnasium import spaces
import numpy as np

"""
The wide representation where the agent can pick the tile position and tile value at each update.
"""

class WideRepresentation(Representation):
    def __init__(self):
        super().__init__()
        self._x = 0
        self._y = 0

    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._x = 0
        self._y = 0

    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([width, height, num_tiles])

    def get_observation_space(self, width, height, num_tiles):
        # Add a channel dimension (e.g., shape=(height, width, 1))
        return spaces.Box(
            low=0,
            high=num_tiles - 1,
            shape=(height, width, 1),
            dtype=np.uint8
        )

    def get_observation(self):
        # Add a channel dimension to the map
        return self._map[..., np.newaxis].copy()

    def update(self, action):
        x, y, tile_value = action
        # Store the current x and y for rendering
        self._x = x
        self._y = y
        # Check if the action modifies the map
        change = self._map[y][x] != tile_value
        if change:
            self._map[y][x] = tile_value
        return int(change), x, y

    def render(self, lvl_image, tile_size, border_size):
        # Draw a marker at the last edited position
        marker = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        for x in range(tile_size):
            marker.putpixel((0, x), (0, 255, 0, 255))  # Green color
            marker.putpixel((1, x), (0, 255, 0, 255))
            marker.putpixel((tile_size - 2, x), (0, 255, 0, 255))
            marker.putpixel((tile_size - 1, x), (0, 255, 0, 255))
        for y in range(tile_size):
            marker.putpixel((y, 0), (0, 255, 0, 255))
            marker.putpixel((y, 1), (0, 255, 0, 255))
            marker.putpixel((y, tile_size - 2), (0, 255, 0, 255))
            marker.putpixel((y, tile_size - 1), (0, 255, 0, 255))
        lvl_image.paste(
            marker,
            (
                (self._x + border_size[0]) * tile_size,
                (self._y + border_size[1]) * tile_size,
                (self._x + border_size[0] + 1) * tile_size,
                (self._y + border_size[1] + 1) * tile_size,
            ),
            marker
        )
        return lvl_image