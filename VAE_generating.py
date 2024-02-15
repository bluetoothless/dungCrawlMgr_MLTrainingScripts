import os
import json
import numpy as np
import onnx
import onnxruntime as ort

VAE_results_directory = "D:/Github/dungeonCrawlerMgr/LevelData/results_VAE/"

def get_last_number():
    files = os.listdir(VAE_results_directory)
    max_num = 0
    for file in files:
        if "VAE_model_" in file:
            try:
                num = int(file.split("_")[-1].split(".")[0])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return max_num

def load_model():
    model_path = os.path.join(VAE_results_directory, f"VAE_model_{get_last_number()}.onnx")
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(model_path)
    #input_names = [input.name for input in ort_session.get_inputs()]
    #print("Input Names:", input_names)
    return ort_session

def generate_map(ort_session, latent_dim):
    # dummy_input = np.random.randn(1, 2800).astype(np.float32) # <- old model
    # recon_map = ort_session.run(None, {'onnx::Reshape_0': dummy_input})[0] # <- old model

    #dummy_input = np.random.randn(1, 7, 18, 18).astype(np.float32) # <- abs/2006.09807
    #recon_map = ort_session.run(None, {'onnx::Reshape_0': dummy_input})[0] # <- abs/2006.09807

    latent_vector = np.random.normal(loc=0.0, scale=1.0, size=(1, latent_dim)).astype(np.float32)
    recon_map = ort_session.run(None, {'onnx::Gemm_0': latent_vector})[0]

    # Reshaping and getting the most probable class for each tile
    map_reshaped = recon_map.reshape(18, 18, 7)
    generated_map = np.argmax(map_reshaped, axis=-1)

    return generated_map.tolist()

def add_edges_to_map(generated_map):
    # Convert list to NumPy array for easier manipulation
    map_array = np.array(generated_map)

    # Create edges
    edge_row = np.ones((map_array.shape[1],), dtype=int)
    edge_col = np.ones((map_array.shape[0] + 2, 1), dtype=int)  # +2 because we're adding rows too

    # Add edges to the map
    map_with_edges = np.vstack([edge_row, map_array, edge_row])  # Top and bottom edges
    map_with_edges = np.hstack([edge_col, map_with_edges, edge_col])  # Left and right edges

    return map_with_edges.tolist()

# Save the map to a JSON file
def save_to_json(data, index):
    filepath = os.path.join(VAE_results_directory, f'generated_levelTileArray_{index}.json')
    data = {
        "LevelTileArray": data
    }
    with open(filepath, 'w') as f:
        json_str = json.dumps(data, separators=(',', ':'))
        formatted_str = json_str.replace('],', '],\n\t\t').replace('[[', '[\n\t\t[').replace(']]', ']\n\t]')
        f.write('{\n\t' + formatted_str[1:-1] + '\n}')

        #json.dump({"LevelTileArray": data}, f)


# Main Execution
if __name__ == "__main__":
    LATENT_DIM = 32
    model = load_model()
    generated_map = generate_map(model, LATENT_DIM)
    generated_map_with_edges = add_edges_to_map(generated_map)
    save_to_json(generated_map_with_edges, get_last_number())