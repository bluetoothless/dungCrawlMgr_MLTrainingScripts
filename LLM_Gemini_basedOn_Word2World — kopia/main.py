import argparse
import google.generativeai as genai
import os
import json
import gemini_generator
from dotenv import load_dotenv
from gemini_generator import GeminiGenerator

LLM_results_directory = "D:/Github/dungeonCrawlerMgr/LevelData/results_LLM/"

class Config:
    model_name = "gemini-1.0-pro"
    story_paragraphs = [4, 5]
    total_objectives = 5
    rounds = 2 # number of rounds to loop over

    experiment_name = "pcgml_LLM" 
    save_dir = f"outputs/{experiment_name}"
    tile_data_dir = "data"

class EnvManager:
    def __init__(self):
        self.total_input_tokens = []
        self.total_output_tokens = []
        self.worlds_history = {}
        self.previous_story = []
        self.previous_tile_map = []
        self.previous_map = []
        self.previous_eval = []
        self.prev_agent_reward = []
        self.total_spent = 0

    def run(self, cfg):
        load_dotenv()
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        gemini_api_key=os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=gemini_api_key)

        #response = model.generate_content("What is the meaning of life?")
        #print(response)
        model = genai.GenerativeModel(cfg.model_name)
        chat = model.start_chat(history=[])
        goals = "finding the most treasure he can, defeating the most enemies he can, avoiding traps and being beaten by enemies which decreases his health, avoiding decreasing the health value to 0 which results in dying, and going the furthest into the dungeon he can by finding portals"
        story = f"A knight explores a dungeon. The dungeon consists of many corridors like a maze and contains skeleton enemies, traps, treasures and teleports that transport to other dungeon levels. The knight has objectives of {goals}."
        generator = GeminiGenerator(model, chat, story, goals, self.total_input_tokens, self.total_output_tokens)
        #print(f"History(0): {generator.chat.history}")

        tile_map_dict = generator.map_tiles_to_chars()
        generator.extract_goals()

        #walkable_tiles_list = generator.extract_walkable_tiles(tile_map_dict)
        walkable_tiles_keys = ['EmptyTile', 'PlayerStartTile', 'PlayerEndTile', 'EnemyTile', 'TreasureTile', 'TrapTile']
        walkable_tiles_list = [tile_map_dict[key] for key in walkable_tiles_keys if key in tile_map_dict]
        important_tiles_keys = ['EmptyTile', 'WallTile']
        important_tiles_list = [tile_map_dict[key] for key in important_tiles_keys if key in tile_map_dict]
        interactive_tiles_keys = ['EnemyTile', 'TreasureTile', 'TrapTile', 'PlayerEndTile']
        interactive_tiles_list = [tile_map_dict[key] for key in interactive_tiles_keys if key in tile_map_dict]
        #print(f"History(1): {generator.chat.history}")
        # cfg.rounds = 1
        for rounds in range(cfg.rounds):
            print(f"ROUND # {rounds}\n")
            world_map_fixed, world_map_fixed_with_chars, world_map_result, world_eval_dict, tileset_used_orig, tileset_used_dict, \
            story_paragraphs, objectives, total_objectives, good_feedback_check, bad_feedback_check, \
            no_of_important_tiles, agent_reward, astar_path= generator.world_generation(rounds,
                                                                                        self.previous_story,
                                                                                        cfg.story_paragraphs,
                                                                                        cfg.total_objectives,
                                                                                        self.previous_tile_map,
                                                                                        self.previous_map,
                                                                                        self.previous_eval,
                                                                                        tile_map_dict, 
                                                                                        important_tiles_list, 
                                                                                        walkable_tiles_list,
                                                                                        interactive_tiles_list,
                                                                                        cfg.save_dir)
            
            self.previous_story.append(story)
            self.previous_tile_map.append(tileset_used_dict)
            self.previous_map.append(world_map_fixed)
            self.previous_eval.append(world_eval_dict)
            self.prev_agent_reward.append(agent_reward)

            self.worlds_history[f"round_{rounds}"] = {"story": story,
                                                        "tile_mapping": tileset_used_dict,
                                                        "goals": goals,
                                                        "objectives": objectives,
                                                        "important_tiles": important_tiles_list,
                                                        "walkable_tiles": walkable_tiles_list,
                                                        "interactive_object_tiles": interactive_tiles_list,
                                                        "world_1st_layer": {"world":world_map_fixed, "tiles": tileset_used_orig},
                                                        "world": world_map_fixed_with_chars,
                                                        "evaluations": world_eval_dict,
                                                        "complexity": {
                                                            "good_feedback_check": good_feedback_check,
                                                            "bad_feedback_check": bad_feedback_check, 
                                                            "no_of_important_tiles": no_of_important_tiles,
                                                            "story_paragraphs": story_paragraphs,
                                                            "total_objectives": total_objectives
                                                        }}
            
            with open(cfg.save_dir +f"/data_gen_{cfg.experiment_name}.json", 'w') as f:
                json.dump(self.worlds_history, f)


            # spent_this_gen = (sum(self.total_input_tokens)/1000)*0.01 + (sum(self.total_output_tokens)/1000)*0.03 
            # self.total_spent += spent_this_gen
            # print(f"$ spent on this gen = {spent_this_gen}")
            # print(f"Total spent = {self.total_spent}")
        print("-------------------------------------------------")
        print(f"Final map:\n{world_map_result}")
        self.save_map(tile_map_dict, world_map_result)

    def get_last_number(self):
        files = os.listdir(LLM_results_directory)
        max_num = 0
        for file in files:
            if "generated_" in file:
                try:
                    num = int(file.split("_")[-1].split(".")[0])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return max_num

    def save_map(self, tile_map_dict, world_map_raw_with_chars):
        symbol_to_index = {v: k for k, v in enumerate(tile_map_dict.values())}
        map_lines = [list(line) for line in world_map_raw_with_chars.strip().split("\n")]
        level_tile_array = [[symbol_to_index[char] for char in line] for line in map_lines]
        index = str(int(self.get_last_number()) + 1)
        filepath = os.path.join(LLM_results_directory, f'generated_levelTileArray_{index}.json')
        data = {
            "LevelTileArray": level_tile_array
        }
        with open(filepath, 'w') as f:
            json_str = json.dumps(data, separators=(',', ':'))
            formatted_str = json_str.replace('],', '],\n\t\t').replace('[[', '[\n\t\t[').replace(']]', ']\n\t]')
            f.write('{\n\t' + formatted_str[1:-1] + '\n}')

def main():
    config = Config()

    parser = argparse.ArgumentParser(description="Process inputs.")
    parser.add_argument('--model', type=str, help='Defaults to "gemini-1.0-pro".')
    parser.add_argument('--min_story_paragraphs', type=int, help='Defaults to "4". Minimum number of paragraphs.')
    parser.add_argument('--max_story_paragraphs', type=int, help='Defaults to "5". Maximum number of paragraphs.')
    parser.add_argument('--total_objectives', type=int, help='Dafualts to 8. Number of objectives in the story.')
    parser.add_argument('--rounds', type=str, help='Defaults to 3. Rounds of world generation.')
    parser.add_argument('--experiment_name', type=str, help='Defaults to "pcgml_LLM".')
    parser.add_argument('--save_dir', type=str, help='Defaults to "outputs/{--experiment_name}"')

    args = parser.parse_args()
    
    if args.model:
        config.model = args.model
    if args.min_story_paragraphs and args.max_story_paragraphs:
        if (args.min_story_paragraphs > args.max_story_paragraphs):
            raise ValueError("Minimum number of paragraphs should be less than maximum number of paragraphs.") 
        if (args.max_story_paragraphs < args.min_story_paragraphs):
            raise ValueError("Maximum number of paragraphs should be greater than minimum number of paragraphs.") 
        config.story_paragraphs = [int(args.min_story_paragraphs),int(args.max_story_paragraphs)]
    if args.total_objectives:
        config.total_objectives = int(args.total_objectives)
    if args.rounds:
        config.rounds = int(args.rounds)
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.save_dir:
        config.save_dir = args.save_dir
    
    env = EnvManager()
    env.run(config)

if __name__ == "__main__":
    main()