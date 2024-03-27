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
            "EmptyTile": 0.4967,
            "WallTile": 0.4361,
            "PlayerStartTile": 0.0031,
            "PlayerEndTile": 0.0031,
            "EnemyTile": 0.0282,
            "TreasureTile": 0.0237,
            "TrapTile": 0.0091
        }
        self._border_tile = "WallTile"

        self._max_treasure = 10
        self._max_enemies = 11
        self._max_traps = 6

        self._target_enemy_dist = 1
        self._target_treasure_dist = 3
        self._target_path = 32

        self._rewards = {
            "PlayerStartTile": 6,
            "PlayerEndTile": 6,
            "TreasureTile": 2,
            "EnemyTile": 2,
            "TrapTile": 2,
            "nearest-enemy-or-trap": 1,
            "too-close-treasure-pairs": 1,
            "regions": 7,
            "path-length": 1,
            #"walls-not-connected": -4
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["EmptyTile", "WallTile", "PlayerStartTile", "PlayerEndTile", "EnemyTile", "TreasureTile", "TrapTile"]

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intialization, the names are "EmptyTile", "WallTile"
        target_path (int): the current path length that the episode turn when it reaches
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)

        self._target_enemy_dist = kwargs.get('target_enemy_dist', self._target_enemy_dist)
        self._target_path = kwargs.get('target_path', self._target_path)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

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
            "too-close-treasure-pairs": 0,
            "regions": calc_num_regions(map, map_locations, ["EmptyTile", "PlayerStartTile", "PlayerEndTile", "EnemyTile", "TreasureTile", "TrapTile"]),
            "path-length": 0
        }
        if map_stats["PlayerStartTile"] == 1 and map_stats["regions"] == 1:
            p_x,p_y = map_locations["PlayerStartTile"][0]
            enemies_and_traps = []
            enemies_and_traps.extend(map_locations["EnemyTile"])
            enemies_and_traps.extend(map_locations["TrapTile"])
            if len(enemies_and_traps) > 0:
                dikjstra,_ = run_dikjstra(p_x, p_y, map, ["EmptyTile", "PlayerStartTile", "EnemyTile", "TrapTile"])
                min_dist = self._width * self._height
                for e_x,e_y in enemies_and_traps:
                    if dikjstra[e_y][e_x] > 0 and dikjstra[e_y][e_x] < min_dist:
                        min_dist = dikjstra[e_y][e_x]
                map_stats["nearest-enemy-or-trap"] = min_dist

            if map_stats["PlayerEndTile"] == 1:
                e_x,e_y = map_locations["PlayerEndTile"][0]
                dikjstra,_ = run_dikjstra(p_x, p_y, map, ["EmptyTile", "PlayerStartTile", "PlayerEndTile", "EnemyTile", "TreasureTile", "TrapTile"])
                map_stats["path-length"] += dikjstra[e_y][e_x]

            treasure_locations = map_locations.get("TreasureTile", [])
            count = 0
            for i, (x1, y1) in enumerate(treasure_locations):
                for j, (x2, y2) in enumerate(treasure_locations):
                    if i < j:  # This ensures each pair is only counted once
                        distance = abs(x2 - x1) + abs(y2 - y1)
                        if distance < self._target_treasure_dist:
                            count += 1
            map_stats["too-close-treasure-pairs"] = count
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
            "PlayerStartTile": get_range_reward(new_stats["PlayerStartTile"], old_stats["PlayerStartTile"], 1, 1),
            "PlayerEndTile": get_range_reward(new_stats["PlayerEndTile"], old_stats["PlayerEndTile"], 1, 1),
            "TreasureTile": get_range_reward(new_stats["TreasureTile"], old_stats["TreasureTile"], self._max_treasure - 4, self._max_treasure),
            "EnemyTile": get_range_reward(new_stats["EnemyTile"], old_stats["EnemyTile"], self._max_enemies - 4, self._max_enemies),
            "TrapTile": get_range_reward(new_stats["TrapTile"], old_stats["TrapTile"], self._max_traps - 4, self._max_traps),
            "nearest-enemy-or-trap": get_range_reward(new_stats["nearest-enemy-or-trap"], old_stats["nearest-enemy-or-trap"], self._target_enemy_dist, np.inf),
            "too-close-treasure-pairs": get_range_reward(new_stats["too-close-treasure-pairs"], old_stats["too-close-treasure-pairs"], 0, 0),

            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["PlayerStartTile"] * self._rewards["PlayerStartTile"] +\
            rewards["PlayerEndTile"] * self._rewards["PlayerEndTile"] +\
            rewards["TreasureTile"] * self._rewards["TreasureTile"] +\
            rewards["EnemyTile"] * self._rewards["EnemyTile"] +\
            rewards["TrapTile"] * self._rewards["TrapTile"] +\
            rewards["nearest-enemy-or-trap"] * self._rewards["nearest-enemy-or-trap"] +\
            rewards["too-close-treasure-pairs"] * self._rewards["too-close-treasure-pairs"] +\
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
        return new_stats["nearest-enemy-or-trap"] >= self._target_enemy_dist and new_stats["path-length"] >= self._target_path
        ## training 1
        # return new_stats["regions"] == 1 \
        #     and new_stats["PlayerStartTile"] == 1 \
        #     and new_stats["PlayerEndTile"] == 1

            #\
            #and new_stats["nearest-enemy-or-trap"] >= self._target_enemy_dist \
            #and new_stats["path-length"] >= 10 \
            #and new_stats["too-close-treasure-pairs"] == 0

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
            "PlayerStartTile": new_stats["PlayerStartTile"],
            "PlayerEndTile": new_stats["PlayerEndTile"],
            "TreasureTile": new_stats["TreasureTile"],
            "EnemyTile": new_stats["EnemyTile"],
            "TrapTile": new_stats["TrapTile"],
            "nearest-enemy-or-trap": new_stats["nearest-enemy-or-trap"],
            "too-close-treasure-pairs": new_stats["too-close-treasure-pairs"],
            "regions": new_stats["regions"],
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
                "EmptyTile": Image.open(os.path.dirname(__file__) + "/quitebrightdungeon/EmptyTile.png").convert('RGBA'),
                "EnemyTile": Image.open(os.path.dirname(__file__) + "/quitebrightdungeon/EnemyTile.png").convert('RGBA'),
                "PlayerEndTile": Image.open(os.path.dirname(__file__) + "/quitebrightdungeon/PlayerEndTile.png").convert('RGBA'),
                "PlayerStartTile": Image.open(os.path.dirname(__file__) + "/quitebrightdungeon/PlayerStartTile.png").convert('RGBA'),
                "TrapTile": Image.open(os.path.dirname(__file__) + "/quitebrightdungeon/TrapTile.png").convert('RGBA'),
                "TreasureTile": Image.open(os.path.dirname(__file__) + "/quitebrightdungeon/TreasureTile.png").convert('RGBA'),
                "WallTile": Image.open(os.path.dirname(__file__) + "/quitebrightdungeon/WallTile.png").convert('RGBA')
            }
        return super().render(map)
