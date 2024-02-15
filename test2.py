# percentages
# 0 (tile type): 53102 [40.23%]
# 1 (tile type): 71706 [54.32%]
# 2 (tile type): 330 [0.25%]
# 3 (tile type): 330 [0.25%]
# 4 (tile type): 3018 [2.29%]
# 5 (tile type): 2537 [1.92%]
# 6 (tile type): 977 [0.76%]

import os
import json
from collections import defaultdict


def analyze_tile_maps(directory):
    tile_count = defaultdict(int)
    tile_percentage_in_map = defaultdict(list)

    for filename in os.listdir(directory):
        if filename.startswith("levelTileArray_") and filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r') as file:
                data = json.load(file)
                level_tiles = data['LevelTileArray']

                total_tiles = sum(len(row) for row in level_tiles)
                current_map_count = defaultdict(int)

                for row in level_tiles:
                    for tile in row:
                        tile_count[tile] += 1
                        current_map_count[tile] += 1

                for tile, count in current_map_count.items():
                    percentage = (count / total_tiles) * 100
                    tile_percentage_in_map[tile].append(percentage)

    average_percentage = {tile: sum(percentages) / len(percentages)
                          for tile, percentages in tile_percentage_in_map.items()}

    result_str = ""
    for tile, count in tile_count.items():
        result_str += f"{tile} (tile type): {count} [{average_percentage[tile]:.2f}%], "

    return result_str


# Use the function with your directory path
directory = "D:/Github/dungeonCrawlerMgr/LevelData/training_data"
print(analyze_tile_maps(directory))
