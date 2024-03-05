"""
Run a trained agent and get generated maps
"""
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs
import json
import numpy as np

def convert_np_array_to_list(obj):
    """
    Recursively convert numpy arrays in the given object to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_array_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_array_to_list(element) for element in obj]
    else:
        return obj

def one_hot_to_digit(one_hot_vector):
    return int(np.argmax(one_hot_vector))
def convert_terminal_observation2(terminal_observation):
    return [[one_hot_to_digit(vector) for vector in row] for row in terminal_observation]
def convert_terminal_observation(terminal_observation):
    return [[sublist.index(1.0) for sublist in outer_list] for outer_list in terminal_observation]

def pad_array(array, pad_width, pad_value=1):
    new_height = len(array) + (2 * pad_width)
    new_width = len(array[0]) + (2 * pad_width)
    padded_array = [[pad_value for _ in range(new_width)] for _ in range(new_height)]
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            padded_array[i + pad_width][j + pad_width] = value
    return padded_array

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
    elif game == "quitebrightdungeon":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
    kwargs['render'] = True

    agent = PPO2.load(model_path)

    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    elif game == "quitebrightdungeon":
        kwargs['cropped_size'] = 18

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()
    obs = env.reset()
    dones = False
    for i in range(kwargs.get('trials', 1)):
        while not dones:
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                print(info[0])
            if dones:
                generated_map = convert_np_array_to_list(info[0])
                results_file = f'generation_results.json'
                with open(results_file, 'w') as f:
                    json.dump(generated_map, f, indent=4)
                # with open(results_file, 'r') as f:
                #     data = json.load(f)
                # level_tile_array = convert_terminal_observation(data['terminal_observation'])
                # for row in level_tile_array:
                #     print(f"{row}")
                # print("\n")
                # level_tile_array = convert_terminal_observation2(data['terminal_observation'])
                # for row in level_tile_array:
                #     print(f"{row}")
                # level_tile_array_padded = pad_array(level_tile_array, 1)
                # data = { "LevelTileArray": level_tile_array_padded }
                # with open('generated_map.json', 'w') as f:
                #     json_str = json.dumps(data, separators=(',', ':'))
                #     formatted_str = json_str.replace('],', '],\n\t\t').replace('[[', '[\n\t\t[').replace(']]', ']\n\t]')
                #     f.write('{\n\t' + formatted_str[1:-1] + '\n}')
                break
        time.sleep(0.2)

################################## MAIN ########################################
game = 'quitebrightdungeon'
representation = 'narrow'
model_path = 'models/{}/{}/model_2.pkl'.format(game, representation)
kwargs = {
    'change_percentage': 0.4,
    'trials': 1,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
