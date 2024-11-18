from gymnasium import spaces
from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
import numpy as np
from collections import OrderedDict

class NarrowRepresentation(Representation):
    def __init__(self):
        super().__init__()
        self._random_tile = True
        self._x = 0
        self._y = 0

    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._x = self._random.randint(width)
        self._y = self._random.randint(height)

    def get_action_space(self, width, height, num_tiles):
        return spaces.Discrete(num_tiles + 1)

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width - 1, height - 1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles - 1, shape=(height, width), dtype=np.uint8)
        })

    def get_observation_space(self, width, height, num_tiles):
        # Combine the position (2D) and map into a single Box
        return spaces.Box(
            low=0,
            high=max(num_tiles - 1, width - 1, height - 1),
            shape=(height, width, 1 + 1),  # Add an extra channel for position
            dtype=np.uint8
        )

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._random_tile = kwargs.get('random_tile', self._random_tile)

    def update(self, action):
        change = 0
        x, y = self._x, self._y

        if action > 0:
            if self._map[y][x] != action - 1:
                change = 1
                self._map[y][x] = action - 1

        if self._random_tile:
            self._x = self._random.randint(self._map.shape[1])
            self._y = self._random.randint(self._map.shape[0])
        else:
            self._x += 1
            if self._x >= self._map.shape[1]:
                self._x = 0
                self._y += 1
                if self._y >= self._map.shape[0]:
                    self._y = 0

        return change, x, y

    def render(self, lvl_image, tile_size, border_size):
        x_graphics = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        for x in range(tile_size):
            x_graphics.putpixel((0, x), (255, 0, 0, 255))
            x_graphics.putpixel((1, x), (255, 0, 0, 255))
            x_graphics.putpixel((tile_size - 2, x), (255, 0, 0, 255))
            x_graphics.putpixel((tile_size - 1, x), (255, 0, 0, 255))
        for y in range(tile_size):
            x_graphics.putpixel((y, 0), (255, 0, 0, 255))
            x_graphics.putpixel((y, 1), (255, 0, 0, 255))
            x_graphics.putpixel((y, tile_size - 2), (255, 0, 0, 255))
            x_graphics.putpixel((y, tile_size - 1), (255, 0, 0, 255))
        lvl_image.paste(
            x_graphics,
            (
                (self._x + border_size[0]) * tile_size,
                (self._y + border_size[1]) * tile_size,
                (self._x + border_size[0] + 1) * tile_size,
                (self._y + border_size[1] + 1) * tile_size,
            ),
            x_graphics
        )
        return lvl_image
