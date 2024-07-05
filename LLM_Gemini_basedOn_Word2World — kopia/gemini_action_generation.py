import time
from solvers import WorldState, EnhancedAStarWorldAgent
from gemini_env import GeminiEnv
import traceback
from utils import (
    extract_dict,
    extract_list,
    pad_rows_to_max_length,
    parse_grid,
    find_most_similar_images,
    bert_batch_similarity,
    find_character_position,
    list_of_lists_to_string
)

class GeminiActionGeneration():
    def __init__(self, model, story):
        self.model = model
        self.story = story
        self.chat = None

    def action_generation(self, round,
                        world_map_fixed,
                        world_map_fixed_with_chars,
                        tileset_used_dict_1st_layer,
                        tileset_used_dict,
                        walkable_tiles_list,
                        object_tiles_list,
                        goal_discriptions, 
                        save_dir):
                
        print("Generating Actions...")
        except_done = False
        NO_OF_EXCEPTIONS_3 = 0
        total_reward = 0
        frames = [] 
        total_episodes = 1
        episodes = 0
        all_episodes_rewards = []
        try:
            while not except_done:
                
                
                objective_tile_system = "You are a great planner in 2D game. You plan objectives for the protagonist of the game. All objectives should match the goals extracted from the story. Objectives should strictly follow them. Return a Python dictionary of the objective as the key and a tile that achieves the objective and the position of the tile. For example 'Objective': ['A', 6, 1]. Only return a Python dictionary. Do not return a python response."
                objective_tile_prompt = f"Given the story:\n{self.story}\n a 2D tile map of a world was created for the story:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping:\n{tileset_used_dict}\n You are also provided with the description of the goals:\n{goal_discriptions}\n and walkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n Taking this information into your context, create the objectives to achieve and also provide the tile that you will pick up, reach the position at or hit the enemy. Return a Python dictionary of the objective as the key and a tile that achieves the objective and the position of the tile. The pattearn should be 'Objective': ['tile', row, column], for example 'Objective': ['A', 6, 1], thus the first element would be the tile and second and third elements of the list will be position of the tile. Return strictly in this format."
                
                self.chat = self.model.start_chat(history=[])
                # messages = [
                #     {"role": "system", "content": objective_tile_system},
                #     {"role": "user", "content": objective_tile_prompt}
                # ]
                # objective_tile_discriptions = self.chat.send_message(messages)    
                _ = self.chat.send_message(objective_tile_system)
                objective_tile_discriptions = self.chat.send_message(objective_tile_prompt)  
                
                print("Objectives: \n")
                print(objective_tile_discriptions._result.candidates[0].content.parts[0].text)
                print("\n")
                
                objective_tile_dict = extract_dict(objective_tile_discriptions._result.candidates[0].content.parts[0].text)

                print("objective_tile in a dict: \n")
                print(objective_tile_dict)
                print("\n")

                total_actions = {}
                objective_flag = False

                # walkable_tiles_list = extract_list(str(walkable_tiles_list))
                # object_tiles_list = extract_list(str(object_tiles_list))

                # ASTAR Search
                world_map_fixed_with_chars = pad_rows_to_max_length(world_map_fixed_with_chars)
                parsed_world_map = parse_grid(world_map_fixed_with_chars)

                objective_tile_list = []
                for _keys, str_obj in objective_tile_dict.items():
                    print(f"str_obj: {str(str_obj)}")
                    temp_list = extract_list(str(str_obj))
                    print(f"temp_list = {temp_list}")
                    if len(temp_list) > 2:
                        objective_tile_list.append((temp_list[1], temp_list[2]))
                
                solving = False
                solving_exceptions = 0
                while not solving:
                    try:
                        # Initialize the game state and agent
                        game_state = WorldState(walkable_tiles_list, object_tiles_list, parsed_world_map, objective_tile_list)
                        game_state = game_state.stringInitialize(parsed_world_map, objective_tile_list)
                        astar_agent = EnhancedAStarWorldAgent(walkable_tiles_list, objective_tile_list, game_state, object_tiles_list)
                        astar_path, _, _, game_map_updated, _ = astar_agent.getSolution(game_state,maxIterations=10000)
                        print(f"astar_path: {len(astar_path)}")
                        solving = True
                    except Exception as e:
                        #print(f"check#3 done = {done}")
                        tb = traceback.format_exc()
                        print(f"Exception raised: {e}\n {tb}")
                        solving_exceptions += 1
                        if solving_exceptions >= 5:
                            solving = True
                            astar_path = []
                            print(f"astar_path: {len(astar_path)}")
                        pass
                    

                removed_value = tileset_used_dict.pop('Protagonist', None) 
                removed_value = tileset_used_dict.pop('Antagonist', None) 
                tileset_used_dict["PlayerStartTile"] = "@"
                tileset_used_dict["EnemyTile"] = "&"

                env = GeminiEnv(walkable_tiles_list, world_map_fixed, world_map_fixed_with_chars, object_tiles_list, "&")
                agent = LLMAgent()
                state = env.reset()

                reward_feedback = "This is your first objective"
                reward_design = {
                    "You are 8 tiles away from objective thus objective is incomplete": -100,
                    "You are 5 to 8 tiles away from objective thus objective is incomplete": -50,
                    "You are 3 to 5 tiles away from objective": +25,
                    "You are 1 to 3 tiles away from objective": +50,
                    "You are 1 tile away or your are on the objective tile from objective": +100,
                    "You have completed the objective": +100,
                }
                protagonist_position = find_character_position(world_map_fixed_with_chars, "@")
                
                prev_episode_reward = 0
                done = False
                
                while not done:
                    reward = 0
                    for i in range(len(objective_tile_dict)):
                        
                        print("\n")
                        print(f"OBJECTIVE: {list(objective_tile_dict.keys())[i]}")
                        print("\n")
                        total_actions[list(objective_tile_dict.keys())[i]] = []
                        #while not objective_flag:
                        action_system = f"You are a great planner in 2D game. You plan actions for the protagonist of the game to achieve all objects. You are given objectives, tiles and the position of tiles to achieve the objectives. You have the following options as actions: 'move_up', move_down, 'move_right', 'move_left', 'pick_object', 'hit_enemy'. Generate a sequence of actions that will achieve the objective. Only return the sequence of actions from the options."
                        action_prompt = f"Given the story:\n{self.story}\n a 2D tile map of a world was created for the story:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\n and walkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Accumulative rewards for all the previous objectives tille now are {reward}. Taking this information into your context, create a sequence of actions for the protagonist to complete the objective: {list(objective_tile_dict.keys())[i]}, which is to reach the tile, 'pick_object' or 'hit_enemy' at tile and position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary. Do not return it in a Python response."
                        
                        self.chat = self.model.start_chat(history=[])
                        # messages = [
                        #     {"role": "system", "content": action_system},
                        #     {"role": "user", "content": action_prompt}
                        # ]
                        # actions_discriptions = self.chat.send_message(messages)
                        _ = self.chat.send_message(action_system)
                        actions_discriptions = self.chat.send_message(action_prompt)

                        action_dict = extract_dict(actions_discriptions._result.candidates[0].content.parts[0].text)
                        print("Action: \n")
                        print(action_dict["action"])
                        print("\n")
                        total_actions[list(objective_tile_dict.keys())[i]].append(action_dict["action"])
                        
                        
                        for action_str in action_dict["action"]:
                            action = agent.action(action_str)
                            state, _r, done, _ = env.step(action)
                            
                            frame = env.render(mode='rgb_array')  # Capture the frame
                            frames.append(frame)  # Append the frame
                            time.sleep(0.01)
                    
                    
                        current_state = list_of_lists_to_string(state)
                        
                        print(current_state)
                        print("\n")
                        
                        check_prompt = f"Given the previous world state:\n{world_map_fixed_with_chars}\n and the updated state that you returned: \n{current_state}\n is the objective {list(objective_tile_dict.keys())[i]} completed? Remember, from the dictionary of objectives, this objective will be completed when you reach tile {list(objective_tile_dict.values())[0]} at position {list(objective_tile_dict.values())[1]} or you are one tile aound this position in any directions. Strictly, only return 'Complete' or 'Incomplete'."
                        
                        # messages = [
                        #     {"role": "assistant", "content": actions_discriptions._result.candidates[0].content.parts[0].text},
                        #     {"role": "user", "content": check_prompt}
                        # ]
                        # check_discriptions = self.chat.send_message(messages)
                        _ = self.chat.send_message(actions_discriptions._result.candidates[0].content.parts[0].text)
                        check_discriptions = self.chat.send_message(check_prompt)
                        
                        world_map_fixed_with_chars = current_state
                                                
                        for k, value in enumerate(objective_tile_dict.values()):
                            if k == i:
                                objective_pos = extract_list(str(value))
                        protagonist_position = find_character_position(world_map_fixed_with_chars, "@")
                        print("\n")
                        print(f"protagonist_position: {protagonist_position}")
                        print(f"objective_position: [{objective_pos[1]},{objective_pos[2]}]")
                        

                        distance_from_objective = (abs(objective_pos[1] - protagonist_position[0]), abs(objective_pos[2] - protagonist_position[1]))
                        print(f"distance from current objective: [{distance_from_objective[0]}, {distance_from_objective[1]}]") 
                        print("\n")
                        reward_feedback = ""
                        reward_feedback = "Your previous objectives reward feedback is: "
                        if (distance_from_objective[0] > 8 or distance_from_objective[1] > 8):
                            reward -= 100
                            reward_feedback += f"You were very far from the objective tile so you were given a regret(negative reward) of -100 points and objective was INCOMPLETE"
                        if (distance_from_objective[0] > 5 and distance_from_objective[0] < 8) or (distance_from_objective[1] > 5 and distance_from_objective[1] < 8):
                            reward -= 50
                            reward_feedback += f"You were far from the objective tile so you were given a regret(negative reward) of -50 points and objective was INCOMPLETE"
                        if (distance_from_objective[0] <= 5 and distance_from_objective[0] > 3) and (distance_from_objective[1] <= 5 and distance_from_objective[1] > 3):
                            reward += 25
                            reward_feedback += f"You were close to the objective tile so you were given a reward of 25 points"
                        if (distance_from_objective[0] < 3 and distance_from_objective[0] > 1) and (distance_from_objective[1] < 3 and distance_from_objective[1] > 1):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"

                        if (distance_from_objective[0] <= 1) and (distance_from_objective[1] > 1 and distance_from_objective[1] <= 5):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"
                        if (distance_from_objective[1] <= 1) and (distance_from_objective[0] > 1 and distance_from_objective[0] <= 5):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"

                        if (distance_from_objective[0] <= 1 and distance_from_objective[1] <= 1) or check_discriptions._result.candidates[0].content.parts[0].text == "Complete":
                            
                            if (distance_from_objective[0] == 0 and distance_from_objective[1] == 0) and check_discriptions._result.candidates[0].content.parts[0].text == "Complete":
                                reward += 200
                                #objective_flag = True
                                reward_feedback += f"You were by the objective tile and you COMPLETED the objective so you were given a reward of 200 points"
                            else:
                                reward += 100
                                reward_feedback += f"You were by the objective tile so you were given a reward of 100 points"
                        print("\n")
                        print(f"EPISODE REWARDS uptill now: {reward}")
                        print("\n")
                    total_reward += reward
                    episodes += 1
                    all_episodes_rewards.append(reward)
                    print("\n")
                    print(f"TOTAL REWARD for EPISODE: {total_reward}")
                    if episodes == total_episodes:
                        done = True
                
                except_done = True
            
        except Exception as e:
            #print(f"check#3 done = {done}")
            tb = traceback.format_exc()
            print(f"Exception raised: {e}\n {tb}")
            NO_OF_EXCEPTIONS_3 += 1
            if NO_OF_EXCEPTIONS_3 >= 5:
                except_done = True
            pass

        print(f"all_episodes_rewards: {all_episodes_rewards}")
        return max(all_episodes_rewards) if all_episodes_rewards != [] else 0, len(astar_path), objective_tile_dict

class LLMAgent:
    def __init__(self):
        pass

    def action(self, action_string):
        if action_string == 'move_up':
            return 0
        if action_string == 'move_down':
            return 1
        if action_string == 'move_left':
            return 2
        if action_string == 'move_right':
            return 3
        if action_string == 'pick_object':
            return 4
        if action_string == 'hit_enemy':
            return 5