import random
#from rembg import remove
import numpy as np

class DiscreteSpace:
    def __init__(self, n):
        self.n = n
    
    def sample(self):
        return random.randint(0, self.n - 1)
    
    def contains(self, x):
        return 0 <= x < self.n

class BoxSpace:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
    
    def sample(self):
        return np.random.randint(low=self.low, high=self.high, size=self.shape, dtype=self.dtype)
    
    def contains(self, x):
        return (x >= self.low).all() and (x <= self.high).all() and x.shape == self.shape and x.dtype == self.dtype


class GeminiEnv():
    def __init__(self, walkable_tiles, str_map_without_chars, str_map, interactive_object_tiles, enemy_tiles):
        str_map = pad_rows_to_max_length(str_map)
        str_map_without_chars = pad_rows_to_max_length(str_map_without_chars)
        
        self.map_str_without_chars = str_map_without_chars.strip().split('\n')
        self.map_str = str_map.strip().split('\n')
        
        self.tile_size = 16
        self.char_tile_size = 16
        self.action_space = DiscreteSpace(6)  # Up, down, left, right, pick, hit
        self.observation_space = BoxSpace(low=0, high=255, shape=(len(self.map_str) * self.tile_size, len(self.map_str[0]) * self.tile_size, 3), dtype=np.uint8)
        self.default_walkable_tile = "B"
        
        self.walkable_tiles = walkable_tiles
        self.interactive_object_tiles = interactive_object_tiles
        self.enemy_tiles = enemy_tiles
        self.picked_objects = []

        # Make second layer transparent

        # for char, image in self.tiles.items():
        #     if char.isalpha():
        #         self.tiles[char] = remove(self.tiles[char])


        # Count the occurrences of each tile in the map
        tile_counts = {}
        for row in self.map_str:
            for tile in row:
                if tile in walkable_tiles:
                    if tile not in tile_counts:
                        tile_counts[tile] = 1
                    else:
                        tile_counts[tile] += 1

        # Determine the most common walkable tile
        if tile_counts:
            self.default_walkable_tile = max(tile_counts, key=tile_counts.get)
        else:
            raise ValueError("No walkable tiles found in the map.")
        

        self.reset()

    def reset(self):
        self.map = [list(row) for row in self.map_str]
        self.map_without_chars = [list(row) for row in self.map_str_without_chars]
        self.grid_width = max(len(row) for row in self.map)
        self.grid_height = len(self.map)
        self.player_pos = self.find_player_position()
        self.current_tile = self.default_walkable_tile  # Default current tile to 'A', change if necessary
        return self.get_state()

    def step(self, action):
        reward = 0
        if action < 4:  # Movement actions
            self.move_player(action)
        elif action == 4:  # Pick action
            reward += self.pick_object()
        elif action == 5:  # Hit action
            reward += self.hit_enemy()

        
        done = False
        info = {}
        return self.get_state(), reward, done, info

    def move_player(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
        dx, dy = moves[action]
        new_row = self.player_pos[0] + dx
        new_col = self.player_pos[1] + dy

        if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]):
            new_tile = self.map[new_row][new_col]
            if new_tile in self.walkable_tiles:
                self.update_player_position(new_row, new_col, new_tile)

    def update_player_position(self, new_row, new_col, new_tile):
        self.map[self.player_pos[0]][self.player_pos[1]] = self.current_tile
        self.player_pos = (new_row, new_col)
        self.current_tile = new_tile
        self.map[new_row][new_col] = '@'

    def pick_object(self):
        reward = 0
        # Check adjacent tiles for interactive objects and pick them if present
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        for dx, dy in adjacent_positions:
            x, y = self.player_pos
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                target_tile = self.map[new_y][new_x]
                if target_tile in self.interactive_object_tiles:
                    print("Picked an object!")
                    self.map[new_y][new_x] = self.default_walkable_tile 
                    reward = 1
                    break  # Exit after picking up one object
        return reward

    def hit_enemy(self):
        reward = 0
        # Check adjacent tiles for enemies and hit them if present
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        for dx, dy in adjacent_positions:
            x, y = self.player_pos
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                target_tile = self.map[new_y][new_x]
                if target_tile in self.enemy_tiles:  # Assuming enemy_tiles is a list of enemy tile identifiers
                    print("Hit an enemy!")
                    self.map[new_y][new_x] = self.default_walkable_tile  # Replace with default or empty tile
                    reward = 5
                    break  # Exit after hitting one enemy
        return reward
    def get_state(self):
        #print(self.map)
        row_lengths = [len(row) for row in self.map]
        assert len(set(row_lengths)) == 1, "Not all rows in the map have the same length"
        return np.array(self.map)

    def render(self, mode='human'):
        return None

    def find_player_position(self):
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    return (i, j)
        return None
    
def pad_rows_to_max_length(text):
    """
    Pads each row in the provided text to make them of the length of the row with maximum length.
    Rows are padded with the last character found in that row.
    """
    # Split the text into lines
    lines = text.strip().split("\n")
    
    # Determine the maximum line length
    max_length = max(len(line) for line in lines)
    
    # Pad each line to the maximum length
    padded_lines = [line + line[-1] * (max_length - len(line)) if line else "" for line in lines]
    
    return "\n".join(padded_lines)