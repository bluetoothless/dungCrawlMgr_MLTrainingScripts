import os
import json
import numpy as np
import onnx
import onnxruntime as ort

RL_results_directory = "D:/Github/dungeonCrawlerMgr/LevelData/results_RL/"

def get_last_number():
    files = os.listdir(RL_results_directory)
    max_num = 0
    for file in files:
        if "ppo_model" in file:
            try:
                num = int(file.split("epoch")[-1].split(".")[0])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return max_num

# D:/Github/dungeonCrawlerMgr/LevelData/results_RL/ppo_model_epoch7.onnx
def generate_map(onnx_model_path):
    onnx_model_path = f"{onnx_model_path}ppo_model_epoch{get_last_number()}.onnx"
    print(onnx_model_path)
    with open(onnx_model_path, 'rb') as f:
        print(f.read(4))
    print(f"Model valid: {onnx.checker.check_model(onnx_model_path)}")
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    input_sample = np.random.random((1, 7, 20, 20)).astype(np.float32)  # Sample input
    generated_map = session.run(None, {input_name: input_sample})[0]
    return generated_map

def save_to_json(data, index):
    filepath = os.path.join(RL_results_directory, f'generated_levelTileArray_{index}.json')
    data = {
        "LevelTileArray": data.tolist()
    }
    with open(filepath, 'w') as f:
        json_str = json.dumps(data, separators=(',', ':'))
        formatted_str = json_str.replace('],', '],\n\t\t').replace('[[', '[\n\t\t[').replace(']]', ']\n\t]')
        f.write('{\n\t' + formatted_str[1:-1] + '\n}')

def main():
    generated_map = generate_map(RL_results_directory)
    save_to_json(generated_map, get_last_number())

if __name__ == "__main__":
    main()