import json
import os

mapping = {
    0: [1, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0, 0],
    4: [0, 0, 0, 0, 1, 0, 0],
    5: [0, 0, 0, 0, 0, 1, 0],
    6: [0, 0, 0, 0, 0, 0, 1]
}
data_directory = "D:/Github/dungeonCrawlerMgr/LevelData/training_data_augmented/"
#preprocessed_data_directory = "D:/Github/dungeonCrawlerMgr/LevelData/training_data_preprocessed/"
preprocessed_data_directory = "D:/Github/dungeonCrawlerMgr/LevelData/training_data_preprocessed_number_values/"


def load_and_preprocess(index, save=True):
    with open(data_directory + f"levelTileArray_{index}.json", "r") as file:
        data = json.load(file)
        level_tile_array = data["LevelTileArray"]
        preprocessed_data = []
        for row in level_tile_array:
            #  converting data into one-hot vectors
            #new_row = [mapping[item] for item in row]
            preprocessed_data.append(row)#new_row)

    if save:
        if not os.path.exists(preprocessed_data_directory):
            os.makedirs(preprocessed_data_directory)

        with open(f"{preprocessed_data_directory}levelTileArray_{index}.json", "w") as out_file:
            out_file.write('{\n\t"LevelTileArray": [\n')
            for row_idx, row in enumerate(preprocessed_data):
                formatted_row = ','.join(json.dumps(item, separators=(',', ':')) for item in row)
                out_file.write(f'\t\t[{formatted_row}]')
                if row_idx < len(preprocessed_data) - 1:
                    out_file.write(',')
                out_file.write('\n')
            out_file.write('\t]\n}')

    return preprocessed_data


def load_all_data(max_files=10000):
    all_data = []
    for index in range(max_files):
        filepath = data_directory + f"levelTileArray_{index}.json"
        if os.path.exists(filepath):
            all_data.append(load_and_preprocess(index))
        else:
            break
    return all_data


load_all_data()