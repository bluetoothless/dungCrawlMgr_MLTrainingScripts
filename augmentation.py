import os
import json

data_directory = "D:/Github/dungeonCrawlerMgr/LevelData/training_data/"
augmented_data_directory = "D:/Github/dungeonCrawlerMgr/LevelData/training_data_augmented/"

# Function to transpose a 2D array
def transpose_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Function to reverse each row (for horizontal flip)
def reverse_rows(matrix):
    return [list(reversed(row)) for row in matrix]

# Function to reverse each column (for vertical flip)
def reverse_columns(matrix):
    return list(reversed(matrix))

if not os.path.exists(augmented_data_directory):
    os.makedirs(augmented_data_directory)

file_counter = 0

for filename in os.listdir(data_directory):
    if filename.endswith(".json"):
        with open(os.path.join(data_directory, filename), 'r') as f:
            data = json.load(f)
            original_matrix = data["LevelTileArray"]

        # Transpose and flip
        horizontal_flip = reverse_rows(original_matrix)
        vertical_flip = reverse_columns(original_matrix)
        both_flip = reverse_columns(horizontal_flip)

        for i, new_matrix in enumerate([original_matrix, horizontal_flip, vertical_flip, both_flip]):
            new_filename = f"levelTileArray_{file_counter + i}.json"
            new_filepath = os.path.join(augmented_data_directory, new_filename)

            with open(new_filepath, 'w') as f:
                json_str = json.dumps({"LevelTileArray": new_matrix}, separators=(',', ':'))
                formatted_str = json_str.replace('],', '],\n\t\t').replace('[[', '[\n\t\t[').replace(']]', ']\n\t]')
                f.write('{\n\t' + formatted_str[1:-1] + '\n}')


        file_counter += 4

print("Data augmentation completed.")