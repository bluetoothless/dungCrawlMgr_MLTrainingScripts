import os
import json
import numpy as np
import onnx
import onnxruntime as ort

GAN_model_directory = "D:/Github/dungeonCrawlerMgr/LevelData/results_GAN/6_epok/"
GAN_results_directory = "D:/Github/dungeonCrawlerMgr/LevelData/results_GAN/"

def get_last_number():
    files = os.listdir(GAN_model_directory)
    max_num = 0
    for file in files:
        if "generator_" in file:
            try:
                num = int(file.split("_")[-1].split(".")[0])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return max_num

def load_model():
    model_path = os.path.join(GAN_model_directory, f"generator_{get_last_number()}.onnx")
    print(model_path)
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(model_path)
    return ort_session

def generate_map(ort_session):
    # dummy_input = np.random.randn(1, 2800).astype(np.float32) # <- old model
    # recon_map = ort_session.run(None, {'onnx::Reshape_0': dummy_input})[0] # <- old model
    dummy_input = np.random.randn(1, 100, 1, 1).astype(np.float32) # <- abs/2006.09807
    recon_map = ort_session.run(None, {'onnx::ConvTranspose_0': dummy_input})[0] # <- abs/2006.09807

    #print(f"Actual shape of recon_map: {recon_map.shape}")
    # Reshaping and getting the most probable class for each tile
    map_reshaped = recon_map.reshape(20, 20, 7)
    generated_map = np.argmax(map_reshaped, axis=-1)

    return generated_map.tolist()


# Save the map to a JSON file
def save_to_json(data, index):
    filepath = os.path.join(GAN_results_directory, f'generated_levelTileArray_{index}.json')
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
    model = load_model()
    generated_map = generate_map(model)
    save_to_json(generated_map, get_last_number())