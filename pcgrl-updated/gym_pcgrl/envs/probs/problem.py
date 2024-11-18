import numpy as np
import gymnasium as gym

from PIL import Image

class Problem:
    """
    The base class for all the problems that can be handled by the interface
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        self._width = 9
        self._height = 9
        tiles = self.get_tile_types()
        self._prob = {}
        for tile in tiles:
            self._prob[tile] = 1.0 / len(tiles)
    
        self._border_size = (1, 1)
        self._border_tile = tiles[0]
        self._tile_size = 16
        self._graphics = None

    def seed(self, seed=None):
        """
        Seeding the used random variable to get the same result. If the seed is None,
        it will seed it with a random start.

        Parameters:
            seed (int): the starting seed, if it is None a random seed number is used.

        Returns:
            int: the used seed (same as input if not None)
        """
        self._random, seed = gym.utils.seeding.np_random(seed)
        return seed

    def reset(self, start_stats):
        """
        Resets the problem to the initial state and saves the start_stats from the starting map.
        Also, it can be used to change values between different environment resets

        Parameters:
            start_stats (dict(string, any)): the first stats of the map
        """
        self._start_stats = start_stats

    def get_tile_types(self):
        """
        Get a list of all the different tile names

        Returns:
            list of str: contains all the tile names
        """
        raise NotImplementedError('get_tile_types is not implemented')

    def adjust_param(self, **kwargs):
        """
        Adjust the parameters for the current problem

        Parameters:
            width (int): change the width of the problem level
            height (int): change the height of the problem level
            probs (dict(string, float)): change the probability of each tile initialization, 
                                        the names are the same as the tile types from get_tile_types
        """
        self._width = kwargs.get('width', self._width)
        self._height = kwargs.get('height', self._height)
        prob = kwargs.get('probs')
        if prob is not None:
            for t in prob:
                if t in self._prob:
                    self._prob[t] = prob[t]

    def get_stats(self, map):
        """
        Get the current stats of the map

        Returns:
            dict(string, any): stats of the current map to be used in the reward, 
                              episode_over, debug_info calculations
        """
        raise NotImplementedError('get_stats is not implemented')

    def get_reward(self, new_stats, old_stats):
        """
        Get the current game reward between two stats

        Parameters:
            new_stats (dict(string, any)): the new stats after taking an action
            old_stats (dict(string, any)): the old stats before taking an action

        Returns:
            float: the current reward due to the change between the old map stats and the new map stats
        """
        raise NotImplementedError('get_reward is not implemented')

    def get_episode_over(self, new_stats, old_stats):
        """
        Uses the stats to check if the problem ended (episode_over) which means reached
        a satisfying quality based on the stats

        Parameters:
            new_stats (dict(string, any)): the new stats after taking an action
            old_stats (dict(string, any)): the old stats before taking an action

        Returns:
            bool: True if the level reached satisfying quality based on the stats and False otherwise
        """
        raise NotImplementedError('get_episode_over is not implemented')

    def get_debug_info(self, new_stats, old_stats):
        """
        Get any debug information need to be printed

        Parameters:
            new_stats (dict(string, any)): the new stats after taking an action
            old_stats (dict(string, any)): the old stats before taking an action

        Returns:
            dict: debug information that can be used to debug what is happening in the problem
        """
        raise NotImplementedError('get_debug_info is not implemented')

    def render(self, map):
        """
        Get an image on how the map will look like for a specific map

        Parameters:
            map (list of list of str): the current game map

        Returns:
            Image: a pillow image on how the map will look like using the problem graphics
        """
        tiles = self.get_tile_types()
        if self._graphics is None:
            self._graphics = {}
            for i, tile in enumerate(tiles):
                color = (int(i * 255 / len(tiles)), int(i * 255 / len(tiles)), int(i * 255 / len(tiles)), 255)
                self._graphics[tile] = Image.new("RGBA", (self._tile_size, self._tile_size), color)
    
        full_width = len(map[0]) + 2 * self._border_size[0]
        full_height = len(map) + 2 * self._border_size[1]
        lvl_image = Image.new("RGBA", (full_width * self._tile_size, full_height * self._tile_size), (0, 0, 0, 255))
    
        # Render borders
        for y in range(full_height):
            for x in range(self._border_size[0]):
                lvl_image.paste(
                    self._graphics[self._border_tile],
                    (x * self._tile_size, y * self._tile_size, (x + 1) * self._tile_size, (y + 1) * self._tile_size)
                )
                lvl_image.paste(
                    self._graphics[self._border_tile],
                    ((full_width - x - 1) * self._tile_size, y * self._tile_size, 
                     (full_width - x) * self._tile_size, (y + 1) * self._tile_size)
                )
    
        for x in range(full_width):
            for y in range(self._border_size[1]):
                lvl_image.paste(
                    self._graphics[self._border_tile],
                    (x * self._tile_size, y * self._tile_size, (x + 1) * self._tile_size, (y + 1) * self._tile_size)
                )
                lvl_image.paste(
                    self._graphics[self._border_tile],
                    (x * self._tile_size, (full_height - y - 1) * self._tile_size, 
                     (x + 1) * self._tile_size, (full_height - y) * self._tile_size)
                )
    
        # Render map tiles
        for y in range(len(map)):
            for x in range(len(map[y])):
                lvl_image.paste(
                    self._graphics[map[y][x]],
                    ((x + self._border_size[0]) * self._tile_size, 
                     (y + self._border_size[1]) * self._tile_size, 
                     (x + self._border_size[0] + 1) * self._tile_size, 
                     (y + self._border_size[1] + 1) * self._tile_size)
                )
        return lvl_image
