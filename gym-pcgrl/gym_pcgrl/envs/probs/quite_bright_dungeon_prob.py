import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_certain_tile, run_dikjstra, calc_longest_path

"""
Generate a fully connected Quite Bright Dungeon level.

Args:
    target_enemy_dist: enemies should be at least this far from the player on spawn
"""
class QuiteBrightDungeonProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 18
        self._height = 18
        self._prob = {
            "EmptyTile": 0.4,
            "WallTile":0.4,
            "PlayerStartTile":0.02,
            "PlayerEndTile": 0.02,
            "EnemyTile": 0.02,
            "TreasureTile": 0.02,
            "TrapTile": 0.02
        }
        self._border_tile = "WallTile"

        self._max_enemies = 9

        self._target_enemy_dist = 2

        self._target_solution = 20

        self._rewards = {
            "player": 3,
            "traps": 3,
            "regions": 5,
            "enemies": 1,
            "nearest-enemy": 2,
            "path-length": 1,
            "dist-win": 0.0,
            "sol-length": 1
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["EmptyTile", "WallTile", "PlayerStartTile", "PlayerEndTile", "EnemyTile", "TreasureTile", "TrapTile"]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "PlayerStartTile": calc_certain_tile(map_locations, ["PlayerStartTile"]),
            "PlayerEndTile": calc_certain_tile(map_locations, ["PlayerEndTile"]),
            "TreasureTile": calc_certain_tile(map_locations, ["TreasureTile"]),
            "EnemyTile": calc_certain_tile(map_locations, ["EnemyTile"]),
            "TrapTile": calc_certain_tile(map_locations, ["TrapTile"]),
            "nearest-enemy-or-trap": 0,

            "regions": calc_num_regions(map, map_locations, ["EmptyTile", "PlayerStartTile", "EnemyTile", "TreasureTile", "TrapTile"]),
            "path-length": 0
        }
        if map_stats["PlayerStartTile"] == 1 and map_stats["PlayerEndTile"] == 1 and map_stats["regions"] == 1:
            p_x,p_y = map_locations["PlayerStartTile"][0]
            enemies_and_traps = []
            enemies_and_traps.extend(map_locations["EnemyTile"])
            enemies_and_traps.extend(map_locations["TrapTile"])
            if len(enemies_and_traps) > 0:
                dikjstra,_ = run_dikjstra(p_x, p_y, map, ["EmptyTile", "PlayerStartTile", "EnemyTile", "TreasureTile", "TrapTile"])
                min_dist = self._width * self._height
                for e_x,e_y in enemies_and_traps:
                    if dikjstra[e_y][e_x] > 0 and dikjstra[e_y][e_x] < min_dist:
                        min_dist = dikjstra[e_y][e_x]
                map_stats["nearest-enemy-or-trap"] = min_dist

            e_x,e_y = map_locations["PlayerEndTile"][0]
            dikjstra,_ = run_dikjstra(p_x, p_y, map, ["EmptyTile", "PlayerStartTile", "PlayerEndTile", "EnemyTile", "TreasureTile", "TrapTile"])
            map_stats["path-length"] += dikjstra[e_y][e_x]

        return map_stats

    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and lesser number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "key": get_range_reward(new_stats["key"], old_stats["key"], 1, 1),
            "door": get_range_reward(new_stats["door"], old_stats["door"], 1, 1),
            "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], 2, self._max_enemies),
            "nearest-enemy": get_range_reward(new_stats["nearest-enemy"], old_stats["nearest-enemy"], self._target_enemy_dist, np.inf),

            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["key"] * self._rewards["key"] +\
            rewards["door"] * self._rewards["door"] +\
            rewards["enemies"] * self._rewards["enemies"] +\
            rewards["nearest-enemy"] * self._rewards["nearest-enemy"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["path-length"] * self._rewards["path-length"]

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats):
        return len(new_stats["solution"]) >= self._target_solution
    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "key": new_stats["key"],
            "door": new_stats["door"],
            "enemies": new_stats["enemies"],
            "regions": new_stats["regions"],
            "nearest-enemy": new_stats["nearest-enemy"],
            "path-length": new_stats["path-length"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/zelda/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/zelda/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/zelda/player.png").convert('RGBA'),
                "key": Image.open(os.path.dirname(__file__) + "/zelda/key.png").convert('RGBA'),
                "door": Image.open(os.path.dirname(__file__) + "/zelda/door.png").convert('RGBA'),
                "spider": Image.open(os.path.dirname(__file__) + "/zelda/spider.png").convert('RGBA'),
                "bat": Image.open(os.path.dirname(__file__) + "/zelda/bat.png").convert('RGBA'),
                "scorpion": Image.open(os.path.dirname(__file__) + "/zelda/scorpion.png").convert('RGBA'),
            }
        return super().render(map)
