from gymnasium import spaces
from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
import numpy as np


class NarrowRepresentation(Representation):
    def __init__(self):
        print("narrow Representation Initialized") 
        super().__init__()
        self._random_tile = False
        self._x = 0
        self._y = 0

    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._x = np.random.randint(width)
        self._y = np.random.randint(height)

    def get_action_space(self, width, height, num_tiles):
        return spaces.Discrete(num_tiles + 1)

    def get_observation_space(self, width, height, num_tiles):
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
        return {
            "pos": np.array([self._x, self._y], dtype=np.uint8),
            "map": self._map.copy()
        }

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._random_tile = kwargs.get('random_tile', self._random_tile)

    def update(self, action):
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
        highlight = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        
        for x in range(tile_size):
            if x < 2 or x >= tile_size - 2:
                for y in range(tile_size):
                    highlight.putpixel((x, y), (255, 0, 0, 255))  # Red color for borders
                    highlight.putpixel((y, x), (255, 0, 0, 255)) 
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