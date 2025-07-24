
from utils import (
    euclidean_distance,
    extract_dict
)
import traceback

class GeminiEvaluator():
    def __init__(self, model, story):
        self.model = model
        self.story = story
        self.chat = None
    
    def find_special_chars(self, grid):
        special_chars = {}
        for y, row in enumerate(grid):
            for x, char in enumerate(row):
                if not char.isalpha():
                    special_chars[char] = (x, y)
        return special_chars
    
    def parse_grid(self, input_str):
        grid = [list(line) for line in input_str.strip().split('\n')]
        return grid

    def find_characters(self, map):
        return self.find_special_chars(self.parse_grid(map))
    
    def find_important_tiles(self, grid, important_tile_list):
        parsed_grid = self.parse_grid(grid)
        imp_tiles = {}
        print(f"parsed_grid: {parsed_grid}")
        for y, row in enumerate(parsed_grid):
            for x, char in enumerate(row):
                if char in important_tile_list:
                    imp_tiles[char] = (x, y)
        return imp_tiles

    def tile_accuracy(self, map, world_eval_dictionary, important_tiles):

        characters = self.find_characters(map)
                
        total_characters = sum(1 for char in important_tiles if not char.isalpha())
            
        if len(characters) > 0:
            world_eval_dictionary["Accuracy_of_characters_placed"] = (len(characters)/total_characters) * 100
        else:
            world_eval_dictionary["Accuracy_of_characters_placed"] = 0
        
        important_tiles_found = self.find_important_tiles(map, important_tiles)
    
        if len(important_tiles_found) > 0:
            world_eval_dictionary["Accuracy_of_important_tiles_placed"] = (len(important_tiles_found)/len(important_tiles)) * 100
        else:
            world_eval_dictionary["Accuracy_of_important_tiles_placed"] = 0

        return world_eval_dictionary

    def euclidean_distance(self, map, previous_maps, world_eval_dictionary):

        euclidean_distance_score = 0
        if len(previous_maps) > 0:
            for previous_map in previous_maps:
                euclidean_distance_score += euclidean_distance(map, previous_map)
            euclidean_distance_score /= len(previous_maps)
        else:
            euclidean_distance_score = 0
        world_eval_dictionary["Average_euclidean_distance"] = euclidean_distance_score

        return world_eval_dictionary
    
    def evaluate_world(self, map, tile_map_dictionary, walkable_tiles, important_tiles, previous_maps):
        print(f"Evaluating World...")    
        no_of_exceptios = 0
        eval_system_prompt = "You are an evaluator of a 2D tilemap world created from a story. You extract meaning from a given story. You are also provided by Python dictionary-like mapping of tiles and characters or alphabets. You evaluate based on how the tiles have been placed and if they match how the story has explained. For example the protagonist (PlayerStartTile) and the portal (PlayerEndTile) should be placed once. If the protagonist and the portal are placed together then it's a bad placement. If they are far apart then it is good. If the world looks aesthetically nice then you give good evaluation, otherwise bad. If the protagonist or portal tiles are not placed then it is bad. All tiles should be placed nicely. Paths should also be evaluated. Eventually each of the following aspect out of 100: Correct Walkable paths, Aesthetics, Placement of environment tiles, Placement of character tiles, Placement of objects that fulfill goals, and Adherence to story. Present the results as only Python dictionary and don't return it in a Python response."
        evaluation_prompt = f"Given the story:\n{self.story}\nAnd the character mapping:{tile_map_dictionary}\nEvaluate the following 2D tilemap World:\n{map}. Adherence to story should be calculated based on the fact that the map adheres to story through the placement of tiles in the 2D tilemap world."
        done = False
        self.chat = self.model.start_chat(history=[])
        while not done:
            try:
                print(f"check#1 done = {done}")
                
                # message_parameters = {
                #     "model": "gemini-1.0-pro",
                #     "prompt": eval_system_prompt,
                #     "temperature": 0.6
                # }
                # _ = self.chat.send_message(**message_parameters)
                # message_parameters["prompt"] = evaluation_prompt
                # world_eval = self.chat.send_message(**message_parameters)    

                _ = self.chat.send_message(eval_system_prompt)
                world_eval = self.chat.send_message(evaluation_prompt)         

                world_eval_dictionary = extract_dict(world_eval._result.candidates[0].content.parts[0].text)    
                world_eval_dictionary = self.tile_accuracy(map, world_eval_dictionary, important_tiles)
                world_eval_dictionary = self.euclidean_distance(map, previous_maps, world_eval_dictionary)

                print("Evaluation: \n", world_eval_dictionary, "\n")
    
                done = True

            except Exception as e:
                tb = traceback.format_exc()
                print(f"Exception raised: {e}\n{tb}")
                no_of_exceptios += 1
                if no_of_exceptios >= 5:
                    done = True
                pass
        
        return world_eval_dictionary