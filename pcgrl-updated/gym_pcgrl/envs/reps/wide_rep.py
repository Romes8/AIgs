from gymnasium import spaces
from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
import numpy as np


class WideRepresentation(Representation):
    def __init__(self):
        super().__init__()
        self._width = 0
        self._height = 0
        self._num_tiles = 0

    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._x = np.random.randint(width)
        self._y = np.random.randint(height)

    def get_action_space(self, width, height, num_tiles):
        self._width = width
        self._height = height
        self._num_tiles = num_tiles
        return spaces.Discrete(width * height * num_tiles)

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "map": spaces.Box(low=0, high=num_tiles - 1, shape=(height, width), dtype=np.uint8)
        })

    def get_observation(self):
        return {"map": self._map.copy()}

    def update(self, action):
        x, y, v = action
        
        tile_value = v % self._num_tiles
        change = int(self._map[y][x] != tile_value)
        self._map[y][x] = tile_value
        
        self._x, self._y = x, y
        
        return change, x, y

    def render(self, lvl_image, tile_size, border_size):
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