from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gymnasium import spaces
import numpy as np

class TurtleRepresentation(Representation):
    def __init__(self):
        super().__init__()
        self._dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._warp = False
        self._x = 0
        self._y = 0

    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._x = np.random.randint(width)
        self._y = np.random.randint(height)

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._warp = kwargs.get('warp', self._warp)

    def get_action_space(self, width, height, num_tiles):
        return spaces.Discrete(len(self._dirs) + num_tiles)

    def get_observation_space(self, width, height, num_tiles):
        # Combine the map and position into a single Box
        return spaces.Box(
            low=0,
            high=max(num_tiles - 1, 1),
            shape=(height, width, 2),  # Two channels: map and position
            dtype=np.uint8
        )

    def get_observation(self):
        height, width = self._map.shape
        pos_layer = np.zeros((height, width), dtype=np.uint8)
        pos_layer[self._y, self._x] = 1  # Mark the turtle's position
        observation = np.stack((self._map, pos_layer), axis=2).astype(np.float32)
        return observation

    def update(self, action):
        change = 0
        if action < len(self._dirs):
            dx, dy = self._dirs[action]
            new_x = self._x + dx
            new_y = self._y + dy

            if self._warp:
                new_x %= self._map.shape[1]
                new_y %= self._map.shape[0]
            else:
                new_x = max(0, min(new_x, self._map.shape[1] - 1))
                new_y = max(0, min(new_y, self._map.shape[0] - 1))

            self._x, self._y = new_x, new_y
        else:
            tile_value = action - len(self._dirs)
            if self._map[self._y][self._x] != tile_value:
                self._map[self._y][self._x] = tile_value
                change = 1
        return change, self._x, self._y

    def render(self, lvl_image, tile_size, border_size):
        # Draw a marker at the turtle's current position
        marker = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        for x in range(tile_size):
            marker.putpixel((0, x), (0, 0, 255, 255))  # Blue color border
            marker.putpixel((1, x), (0, 0, 255, 255))
            marker.putpixel((tile_size - 2, x), (0, 0, 255, 255))
            marker.putpixel((tile_size - 1, x), (0, 0, 255, 255))
        for y in range(tile_size):
            marker.putpixel((y, 0), (0, 0, 255, 255))
            marker.putpixel((y, 1), (0, 0, 255, 255))
            marker.putpixel((y, tile_size - 2), (0, 0, 255, 255))
            marker.putpixel((y, tile_size - 1), (0, 0, 255, 255))

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
