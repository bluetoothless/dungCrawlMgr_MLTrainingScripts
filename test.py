import json
import os

# Directory paths
input_dir = "D:\\Github\\dungeonCrawlerMgr\\LevelData\\training_data_preprocessed_only2tileTypes"
output_dir = "D:\\Github\\dungeonCrawlerMgr\\LevelData\\training_data_preprocessed_only2tileTypes"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

#tiles_to_preserve = [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]

# Process each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), 'r') as file:
            data = json.load(file)
            level_array = data["LevelTileArray"]

            # Modify the tiles based on the condition
            # for row in level_array:
            #     for i, tile in enumerate(row):
            #         if tile not in tiles_to_preserve:
            #             row[i] = [1, 0, 0, 0, 0, 0, 0]
            for row in level_array:
                for i, tile in enumerate(row):
                    if tile == [1, 0, 0, 0, 0, 0, 0]:
                        row[i] = [1, 0]
                    elif tile == [0, 1, 0, 0, 0, 0, 0]:
                        row[i] = [0, 1]
                    else:
                        print(f"error!!!! Row:{row}, i:{i}, tile:{tile}")

            # Save modified array
            output_filename = os.path.join(output_dir, filename)
            with open(output_filename, 'w') as q_file:
                q_file.write('{\n\t"LevelTileArray": [\n')
                for row_idx, row in enumerate(level_array):
                    formatted_row = ','.join(json.dumps(item, separators=(',', ':')) for item in row)
                    q_file.write(f'\t\t[{formatted_row}]')
                    if row_idx < len(level_array) - 1:
                        q_file.write(',')
                    q_file.write('\n')
                q_file.write('\t]\n}')

print("Processing complete.")