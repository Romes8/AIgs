import os
from PIL import Image
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.probs.sokoban.engine import State, BFSAgent, AStarAgent

class SokobanProblem(Problem):
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
        self._target_solution = kwargs.get('min_solution', self._target_solution)

        rewards = kwargs.get('rewards')
        if rewards:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    def _run_game(self, map):
        game_characters = " #@$."
        string_to_char = {s: game_characters[i] for i, s in enumerate(self.get_tile_types())}
        
        lvl_string = "#" * (self._width + 2) + "\n"
        for row in map:
            lvl_string += "#" + "".join(string_to_char[tile] for tile in row) + "#\n"
        lvl_string += "#" * (self._width + 2)

        state = State()
        state.stringInitialize(lvl_string.split("\n"))

        agents = [BFSAgent(), AStarAgent(), AStarAgent(), AStarAgent()]
        solutions = [agent.getSolution(state, *args) for agent, args in zip(agents, [self._solver_power, (1, self._solver_power), (0.5, self._solver_power), (0, self._solver_power)])]

        for sol, sol_state, _ in solutions:
            if sol_state.checkWin():
                return 0, sol
        return sol_state.getHeuristic(), []

    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "crate": calc_certain_tile(map_locations, ["crate"]),
            "target": calc_certain_tile(map_locations, ["target"]),
            "regions": calc_num_regions(map, map_locations, ["empty", "player", "crate", "target"]),
            "dist-win": self._width * self._height * (self._width + self._height),
            "solution": []
        }
        
        if map_stats["player"] == 1 and map_stats["crate"] == map_stats["target"] > 0 and map_stats["regions"] == 1:
            map_stats["dist-win"], map_stats["solution"] = self._run_game(map)
        return map_stats

    def get_reward(self, new_stats, old_stats):
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

    def render(self, map):
        if self._graphics is None:
            tile_images = ["empty.png", "solid.png", "player.png", "crate.png", "target.png"]
            tile_types = ["empty", "solid", "player", "crate", "target"]
            self._graphics = {
                tile: Image.open(os.path.join(os.path.dirname(__file__), "sokoban", tile)).convert('RGBA')
                for tile, img in zip(tile_types, tile_images)
            }
        return super().render(map)
