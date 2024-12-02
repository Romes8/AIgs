import os
from PIL import Image
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.probs.sokoban.engine import State, BFSAgent, AStarAgent


class SokobanProblem(Problem):
    """
    Sokoban problem generator to create solvable levels with specified parameters.
    """

    def __init__(self):
        super().__init__()
        self._width = 5
        self._height = 5
        self._prob = {"empty": 0.45, "solid": 0.4, "player": 0.05, "crate": 0.05, "target": 0.05}
        self._border_tile = "solid"
        self._solver_power = 5000
        self._max_crates = 3
        self._target_solution = 18
        self._rewards = {
            "player": 3,
            "crate": 2,
            "target": 2,
            "regions": 5,
            "ratio": 2,
            "dist-win": 0.0,
            "sol-length": 1
        }

    def get_tile_types(self):
        return ["empty", "solid", "player", "crate", "target"]

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._solver_power = kwargs.get('solver_power', self._solver_power)
        self._max_crates = kwargs.get('max_crates', self._max_crates)
        self._max_crates = kwargs.get('max_targets', self._max_crates)
        self._target_solution = kwargs.get('min_solution', self._target_solution)
        rewards = kwargs.get('rewards')
        if rewards:
            for key, value in rewards.items():
                if key in self._rewards:
                    self._rewards[key] = value

    def _run_game(self, map_):
        """
        Runs the Sokoban game solver on the input map.
        """
        tile_mapping = {"empty": " ", "solid": "#", "player": "@", "crate": "$", "target": "."}
        lvl_string = "#" * (self._width + 2) + "\n"
        for row in map_:
            lvl_string += "#" + "".join(tile_mapping[tile] for tile in row) + "#\n"
        lvl_string += "#" * (self._width + 2)

        state = State()
        state.stringInitialize(lvl_string.split("\n"))

        agents = [BFSAgent(), AStarAgent()]
        for agent in agents:
            for heuristic in [1, 0.5, 0]:
                # Call getSolution without the solver_power argument
                sol, sol_state, _ = agent.getSolution(state, maxIterations=self._solver_power)  # Adjusted here
                if sol_state.checkWin():
                    return 0, sol
        return state.getHeuristic(), []

    def get_stats(self, map_):
        """
        Computes statistics for the map.
        """
        map_locations = get_tile_locations(map_, self.get_tile_types())
        stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "crate": calc_certain_tile(map_locations, ["crate"]),
            "target": calc_certain_tile(map_locations, ["target"]),
            "regions": calc_num_regions(map_, map_locations, ["empty", "player", "crate", "target"]),
            "dist-win": self._width * self._height * (self._width + self._height),
            "solution": []
        }
        if stats["player"] == 1 and stats["crate"] == stats["target"] and stats["crate"] > 0 and stats["regions"] == 1:
            stats["dist-win"], stats["solution"] = self._run_game(map_)
        return stats

    def get_reward(self, new_stats, old_stats):
        """
        Computes the reward based on changes in map statistics.
        """
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "crate": get_range_reward(new_stats["crate"], old_stats["crate"], 1, self._max_crates),
            "target": get_range_reward(new_stats["target"], old_stats["target"], 1, self._max_crates),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "ratio": get_range_reward(abs(new_stats["crate"] - new_stats["target"]), abs(old_stats["crate"] - old_stats["target"]), -np.inf, -np.inf),
            "dist-win": get_range_reward(new_stats["dist-win"], old_stats["dist-win"], -np.inf, -np.inf),
            "sol-length": get_range_reward(len(new_stats["solution"]), len(old_stats["solution"]), np.inf, np.inf)
        }
        return sum(rewards[key] * self._rewards[key] for key in rewards)

    def get_episode_over(self, new_stats, old_stats):
        return len(new_stats["solution"]) >= self._target_solution

    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "crate": new_stats["crate"],
            "target": new_stats["target"],
            "regions": new_stats["regions"],
            "dist-win": new_stats["dist-win"],
            "sol-length": len(new_stats["solution"])
        }

    def render(self, map_):
        if self._graphics is None:
            self._graphics = {
                "empty": Image.open(os.path.join(os.path.dirname(__file__), "sokoban/empty.png")).convert('RGBA'),
                "solid": Image.open(os.path.join(os.path.dirname(__file__), "sokoban/solid.png")).convert('RGBA'),
                "player": Image.open(os.path.join(os.path.dirname(__file__), "sokoban/player.png")).convert('RGBA'),
                "crate": Image.open(os.path.join(os.path.dirname(__file__), "sokoban/crate.png")).convert('RGBA'),
                "target": Image.open(os.path.join(os.path.dirname(__file__), "sokoban/target.png")).convert('RGBA')
            }
        return super().render(map_)
