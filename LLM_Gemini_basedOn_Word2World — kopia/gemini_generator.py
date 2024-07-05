from gemini_evaluator import GeminiEvaluator
from gemini_action_generation import GeminiActionGeneration
import traceback
from utils import (
    extract_dict,
    extract_list,
    extract_between_ticks,
    remove_extra_special_chars,
    extract_present_elements
)

class GeminiGenerator():
    def __init__(self, model, chat, story, goals, total_input_tokens, total_output_tokens):
        self.model = model
        self.chat = chat
        self.story = story
        self.goals = goals
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens

    def create_story(self, story_paragraphs, total_objectives):
        pass

    def map_tiles_to_chars(self):
        print("Mapping tiles to characters...")
        tileset_map_prompt = f'''Story: "{self.story}"

Let's use the above story to create a 2D game. The exhaustive list of tiles needed to create the environment are: 
EmptyTile (represents walkable empty space),
WallTile (represents walls of the dungeon),
PlayerStartTile (represents the starting point of the player, the knight),
PlayerEndTile (represents the ending point of a level, a portal),
EnemyTile (represents a skeleton enemy),
TreasureTile (represents a treasure),
TrapTile (represents a trap that deals damage when stepped on).

Imagine each tile maps to an alphabet or a character. For environment use alphabets and for characters use special characters. Create it in a single Python Dictionary style. Return only and only a Python Dictionary and nothing else in your response. Don't return it in a Python response. Names should be the Keys and alphabets or characters should be the Values. Protagonist (PlayerStartTile) should always strictly be '@' and the antagonist (EnemyTile) should always strictly be '&'."
'''
        #response = model.generate_content(tileset_map_prompt)
        response = self.chat.send_message(tileset_map_prompt)
        #print(response)
        print("\n")
        #to_markdown(response.text)
        tile_map_dict = extract_dict(response._result.candidates[0].content.parts[0].text)
        print(tile_map_dict)
        print("\n")

        return tile_map_dict

    def extract_goals(self):
        print("Extracting goals...")
        goal_prompt = "What is the main goal for protagonist of the story? What are the small goals for protagonist to achieve the main goal of the story? Also create rewards and penalties based on the goals for protagonist. Create a score for each reward or penalty"
        response = self.chat.send_message(goal_prompt)
        print("\n")
        print(response._result.candidates[0].content.parts[0].text)
        print("\n")

    # def extract_walkable_tiles(self, tile_map_dict, walkable_tiles_list):
    #     print("Extracting walkable tiles...")
    #     walkable_tile_prompt = f"This is the list of the walkable tiles in the 2D tilemap world: \n{walkable_tiles_list}\n"
    #     response = self.chat.send_message(walkable_tile_prompt)
    #     print(response)
    #     print("\n")
    #     walkable_tiles_list = extract_list(response._result.candidates[0].content.parts[0].text)
    #     print(walkable_tiles_list)
    #     print("\n")

    #     return walkable_tiles_list

    def world_generation(self, rounds, previous_story, story_paragraphs, total_objectives, previous_tile_map, previous_map, previous_eval, tile_map_dict, important_tiles_list, walkable_tiles_list, interactive_tiles_list, save_dir):
        print(f"Generating World...")
        NO_OF_EXCEPTIONS_2 = 0
        no_of_important_tiles = 2
        history_to_keep = 0
        good_feedback_prompt = ""
        bad_feedback_prompt = ""
        good_feedback_check = 0
        bad_feedback_check = 0
        if rounds == 0 or history_to_keep > 0:
            world_system_prompt = "You are a 2D game designer that is profficient in designing tile-based maps. Designing any size of the tile-based map is not a problem for you. This is your first round of generation. You are given the goals to achieve and a list of important tiles to place. Consider them to make the world. Do not place the protagonist, the antagonists and the interactive objects of the story right now. Only create the world right now, consisting of empty spaces and walls. Also, consider goals that you extracted earlier and generate while keeping them in context."    
            world_prompt = f"Using the following tile to character mapping:\n{tile_map_dict}\nCreate an entire world on a tile-based grid. Do not create things that would need more than one tile. For example, a house or a building needs more than one tile to be made. Also, following characters are important to place:\n{important_tiles_list}\n and walkable tiles:\n{walkable_tiles_list}\n, and these are interactive objects:\n{interactive_tiles_list}\n. Use {no_of_important_tiles} important tiles to create the world. Do not place the protagonist, the antagonists, and the interactive objects of the story right now. Only create the world right now. It should have 20x20 tiles, so exactly 400 tiles. Create it is a string format with three backticks to start and end with (```) and not in a list format. Make sure that every row has the same number of tiles and that the whole map is 20 by 20 and only has 400 tiles."
        else:
            if len(previous_map) > history_to_keep:
                previous_map = previous_map[-history_to_keep:]
                previous_story = previous_story[-history_to_keep:]
                previous_tile_map = previous_tile_map[-history_to_keep:]
                previous_eval = previous_eval[-history_to_keep:]
            history_intro = f"For your reference, here are the previous {len(previous_map)} stories, their tile mapping and corresponding 2D world maps\n"
            for i in range(len(previous_map)):
                history = history_intro + f"Story {i}: {previous_story[i]}\nTile Map for story {i}:\n{previous_tile_map[i]}\n, 2D World map for story {i}:\n {previous_map[i]} and evaluation for the 2D World map:\n{previous_eval[i]}. {good_feedback_prompt}\n{bad_feedback_prompt} Create higher quality and with a higher diversity map."
            world_system_prompt = f"You are a 2D game designer that is profficient in designing 20x20 tile-based maps. Designing a tile-based map of size 20 by 20 is not a problem for you. This is the generation number {round} for you and you will be provided by previous generation results. Improve evaluation scores in each generation. Previous evaluation scores will be provided to you. You are given the goals to achieve and a list of important tiles to place. Additionally you are given 2D tile-maps and stories that were create before for you to make a better map. Consider them to make the world. Do not place the protagonist, the antagonists, the portals and the important objects of the story right now. Only create the world right now. Also, consider goals that you extracted earlier and generate while keeping them in context."    
            world_prompt = f"Using the following tile to character mapping:\n{tile_map_dict}\nCreate an entire world on a tile-based grid. Do not create things that would neew more than one tile. For example, a house or a building needs more than one tile to be made. Also, following characters are important to place:\n{important_tiles_list}\n and walkable tiles:\n{walkable_tiles_list}\n, and these are interactive objects:\n{interactive_tiles_list}\n. Use {no_of_important_tiles} important tiles to create the world. Do not place the protagonist, the antagonists, and the important objects of the story right now. Only create the world right now. It should have 20x20 tiles, so exactly 400 tiles. Create it is a string format with three backticks to start and end with (```) and not in a list format. Make sure that every row has the same number of tiles and that the whole map is 20 by 20 and only has 400 tiles. {history}"
        done = False
        world_map_fixed = None
        world_map_fixed_with_chars = None
        world_eval_dict = None
        used_char_dict = None
        used_char_dict_with_char = None
        objectives = None
        llm_agent_reward = None
        astar_path = None
        world_map_result = ""
        while not done:
            try:
                _ = self.chat.send_message(world_system_prompt)
                world_response = self.chat.send_message(world_prompt) 

                #print("World: \n")
                #print(f"World system response:\n{world_system_response}\n")
                print(f"World response:\n{world_response}\n")
                print(world_response._result.candidates[0].content.parts[0].text)
                print("\n")
                print("Extracting tilemap...")
                print("\n")

                world_map_raw = extract_between_ticks(world_response._result.candidates[0].content.parts[0].text)
                print(world_map_raw)
                print("\n")
                print("Fixing tilemap...")
                world_map_raw = world_map_raw.replace(' ', '').replace('"', '')
                world_map_fixed = remove_extra_special_chars(world_map_raw)
               
                print(world_map_fixed)
                
                used_char_dict = extract_present_elements(world_map_fixed, tile_map_dict)

                world_with_characters_prompt = f"Now that you have created the following world map:\n{world_map_fixed}\n Place only the protagonist, the antagonists (enemies), the portal and the interactive objects of the story. Do not change anything in the world, just place only the protagonist, the antagonists (enemies), the portal and the interactive objects in the world. Make sure that all of tile types were used. Placing the protagonist, the antagonists (enemies), the portal and the interactive objects should not make the map bigger, instead a placed tile should replace an already existing tile, so if the new tile type is to be placed in mth column in nth row, then the one tile in mth column and nth row should be removed and in its place the new tile type should be added."
                world_with_characters_response = self.chat.send_message(world_with_characters_prompt)

                print("World: \n")
                print(world_with_characters_response._result.candidates[0].content.parts[0].text)
                print("\n")
                print("Extracting tilemap..")
                print("\n")
                world_map_raw_with_chars = extract_between_ticks(world_with_characters_response._result.candidates[0].content.parts[0].text)
                world_map_result = world_map_raw_with_chars
                print(world_map_raw_with_chars)
                print("\n")
                print("Fixing tilemap..")
                world_map_raw_with_chars = world_map_raw_with_chars.replace(' ', '').replace('"', '')
                world_map_fixed_with_chars = remove_extra_special_chars(world_map_raw_with_chars)
                print(world_map_fixed_with_chars)
                
                used_char_dict_with_char = extract_present_elements(world_map_fixed_with_chars, tile_map_dict)

                evaluator = GeminiEvaluator(self.model, self.story)
                world_eval_dict = evaluator.evaluate_world(map=world_map_fixed_with_chars,
                                                            tile_map_dictionary=used_char_dict_with_char,
                                                            walkable_tiles=walkable_tiles_list,
                                                            important_tiles=important_tiles_list,
                                                            previous_maps=previous_map)
                
                action_generation = GeminiActionGeneration(self.model, self.story)
                llm_agent_reward, astar_path, objectives = action_generation.action_generation(rounds, world_map_fixed, world_map_fixed_with_chars, \
                        used_char_dict, used_char_dict_with_char, walkable_tiles_list, important_tiles_list, self.goals, save_dir)
                
                #agent_reward = -1000
                world_eval_dict["agent_reward"] = llm_agent_reward
                world_eval_dict["astar_path"] = astar_path

                story_paragraphs, total_objectives, no_of_important_tiles, bad_feedback_prompt, good_feedback_prompt = self.feedback_checks(rounds, world_eval_dict, previous_eval, story_paragraphs, total_objectives, no_of_important_tiles)
                    
                done = True

            except Exception as e:
                
                tb = traceback.format_exc()
                print(f"Exception raised: {e}\n {tb}")
                NO_OF_EXCEPTIONS_2 += 1

                if NO_OF_EXCEPTIONS_2 >= 2:
                    done = True
                pass

        return world_map_fixed, world_map_fixed_with_chars, world_map_result, world_eval_dict, used_char_dict, used_char_dict_with_char, \
            story_paragraphs, objectives, total_objectives, good_feedback_check, bad_feedback_check, no_of_important_tiles, llm_agent_reward, astar_path


    def feedback_checks(self, rounds, world_eval_dict, previous_eval, story_paragraphs, total_objectives, no_of_important_tiles):
        
        good_feedback_prompt = f"Also, the following is a more detailed feedback of how much you improved in the last generation:\n Your last generation improved the following evaluation metrics, and so you are doing great:\n"
        bad_feedback_prompt = f"\nYour last generation did not improve the following evaluation metrics, and so you need to improve it by being more careful about it:\n"
        good_feedback_check = 0
        bad_feedback_check = 0

        if rounds > 0:
            for key, value in world_eval_dict.items():
                print(f"This round eval: {key}, {value}")
                print(f"Previous round eval: {previous_eval[len(previous_eval) - 1][key]}")

                if key == "astar_path" and world_eval_dict[key] == 0:
                    bad_feedback_check +=2

                if int(world_eval_dict[key]) > int(previous_eval[len(previous_eval) - 1][key]):
                    if key == "agent_reward":
                        good_feedback_check += 2
                    else:
                        good_feedback_check += 1
                    good_feedback_prompt += f"- {key}\n"
                    
                else:
                    if key == "agent_reward":
                        bad_feedback_check += 1
                    else:
                        bad_feedback_check += 1
                    bad_feedback_prompt += f"- {key}\n"
                    
         
            if good_feedback_check == 0:
                good_feedback_prompt = ""
            if bad_feedback_check == 0:
                bad_feedback_prompt = ""
            
            if good_feedback_check >= bad_feedback_check:
                no_of_important_tiles += 1
                story_paragraphs[0] += 1
                story_paragraphs[1] += 1
                total_objectives += 1

        return story_paragraphs, total_objectives, no_of_important_tiles, bad_feedback_prompt, good_feedback_prompt