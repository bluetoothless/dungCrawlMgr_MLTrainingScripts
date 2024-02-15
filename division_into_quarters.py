import json
import os

# Directory paths
input_dir = "D:\\Github\\dungeonCrawlerMgr\\LevelData\\validation_data_preprocessed"
output_dir = "D:\\Github\\dungeonCrawlerMgr\\LevelData\\validation_data_preprocessed_small"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to split map into quarters
def split_map(map_data):
    quarters = []
    mid = len(map_data) // 2
    # Top left
    quarters.append([row[:mid] for row in map_data[:mid]])
    # Top right
    quarters.append([row[mid:] for row in map_data[:mid]])
    # Bottom left
    quarters.append([row[:mid] for row in map_data[mid:]])
    # Bottom right
    quarters.append([row[mid:] for row in map_data[mid:]])
    return quarters

# Process each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), 'r') as file:
            data = json.load(file)
            level_array = data["LevelTileArray"]

            # Remove first and last rows and columns
            modified_array = [row[1:-1] for row in level_array[1:-1]]

            # Split into quarters
            quarters = split_map(modified_array)

            # Save each quarter
            base_name = filename.split('.')[0]
            for i, quarter in enumerate(quarters):
                quarter_filename = f"{base_name}{chr(97+i)}.json"
                with open(os.path.join(output_dir, quarter_filename), 'w') as q_file:
                    q_file.write('{\n\t"LevelTileArray": [\n')
                    for row_idx, row in enumerate(quarter):
                        formatted_row = ','.join(json.dumps(item, separators=(',', ':')) for item in row)
                        q_file.write(f'\t\t[{formatted_row}]')
                        if row_idx < len(quarter) - 1:
                            q_file.write(',')
                        q_file.write('\n')
                    q_file.write('\t]\n}')

print("Processing complete.")
