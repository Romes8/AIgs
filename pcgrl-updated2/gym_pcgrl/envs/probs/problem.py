from gymnasium.utils import seeding
from PIL import Image


class Problem:
    """
    Base class for all problems in the PCGRL environment.
    """
    def __init__(self):
        self._width = 9
        self._height = 9
        tiles = self.get_tile_types()
        self._prob = [1.0 / len(tiles) for _ in tiles]
        self._border_size = (1, 1)
        self._border_tile = tiles[0]
        self._tile_size = 16
        self._graphics = None

    def seed(self, seed=None):
        """
        Sets the random seed for reproducibility.
        """
        self._random, seed = seeding.np_random(seed)
        return seed

    def reset(self, start_stats):
        """
        Resets the problem state, initializing start stats.
        """
        self._start_stats = start_stats

    def get_tile_types(self):
        """
        Returns the list of tile types for the problem.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("get_tile_types is not implemented")

    def adjust_param(self, **kwargs):
        """
        Adjusts problem parameters, such as dimensions and tile probabilities.
        """
        self._width = kwargs.get("width", self._width)
        self._height = kwargs.get("height", self._height)
        prob = kwargs.get("probs")
        if prob:
            for tile, value in prob.items():
                if tile in self.get_tile_types():
                    self._prob[self.get_tile_types().index(tile)] = value

    def get_stats(self, map_):
        """
        Computes statistics for the current map.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("get_stats is not implemented")

    def get_reward(self, new_stats, old_stats):
        """
        Computes the reward based on the change in map statistics.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("get_reward is not implemented")

    def get_episode_over(self, new_stats, old_stats):
        """
        Determines if the episode should terminate based on map statistics.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("get_episode_over is not implemented")

    def get_debug_info(self, new_stats, old_stats):
        """
        Returns debug information for the problem.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("get_debug_info is not implemented")

    def render(self, map_):
        """
        Renders the map as an image.
        """
        tiles = self.get_tile_types()
        if self._graphics is None:
            self._graphics = {
                tile: Image.new(
                    "RGBA", (self._tile_size, self._tile_size),
                    (int(i * 255 / len(tiles)), int(i * 255 / len(tiles)), int(i * 255 / len(tiles)), 255)
                )
                for i, tile in enumerate(tiles)
            }

        full_width = len(map_[0]) + 2 * self._border_size[0]
        full_height = len(map_) + 2 * self._border_size[1]
        lvl_image = Image.new(
            "RGBA",
            (full_width * self._tile_size, full_height * self._tile_size),
            (0, 0, 0, 255),
        )

        # Draw borders
        for y in range(full_height):
            for x in range(self._border_size[0]):
                lvl_image.paste(
                    self._graphics[self._border_tile],
                    (x * self._tile_size, y * self._tile_size, (x + 1) * self._tile_size, (y + 1) * self._tile_size),
                )
                lvl_image.paste(
                    self._graphics[self._border_tile],
                    ((full_width - x - 1) * self._tile_size, y * self._tile_size,
                     (full_width - x) * self._tile_size, (y + 1) * self._tile_size),
                )

        for x in range(full_width):
            for y in range(self._border_size[1]):
                lvl_image.paste(
                    self._graphics[self._border_tile],
                    (x * self._tile_size, y * self._tile_size, (x + 1) * self._tile_size, (y + 1) * self._tile_size),
                )
                lvl_image.paste(
                    self._graphics[self._border_tile],
                    (x * self._tile_size, (full_height - y - 1) * self._tile_size,
                     (x + 1) * self._tile_size, (full_height - y) * self._tile_size),
                )

        # Draw map tiles
        for y, row in enumerate(map_):
            for x, tile in enumerate(row):
                lvl_image.paste(
                    self._graphics[tile],
                    ((x + self._border_size[0]) * self._tile_size,
                     (y + self._border_size[1]) * self._tile_size,
                     (x + self._border_size[0] + 1) * self._tile_size,
                     (y + self._border_size[1] + 1) * self._tile_size),
                )

        return lvl_image
