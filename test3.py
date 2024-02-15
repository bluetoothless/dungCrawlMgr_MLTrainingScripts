import json
import os

# Directory paths
input_dir = "D:\\Github\\dungeonCrawlerMgr\\LevelData\\validation_data_preprocessed_only2tileTypes"
output_dir = "D:\\Github\\dungeonCrawlerMgr\\LevelData\\validation_data_preprocessed_only2tileTypes"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), 'r') as file:
            data = json.load(file)
            level_array = data["LevelTileArray"]

            # Remove first and last rows and columns
            modified_array = [row[1:-1] for row in level_array[1:-1]]

            # Save modified array
            output_filename = os.path.join(output_dir, filename)
            with open(output_filename, 'w') as q_file:
                q_file.write('{\n\t"LevelTileArray": [\n')
                for row_idx, row in enumerate(modified_array):
                    formatted_row = ','.join(json.dumps(item, separators=(',', ':')) for item in row)
                    q_file.write(f'\t\t[{formatted_row}]')
                    if row_idx < len(modified_array) - 1:
                        q_file.write(',')
                    q_file.write('\n')
                q_file.write('\t]\n}')

print("Processing complete.")