import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(low=0, high=5, shape=(grid_size, grid_size), dtype=np.int8)
        
        self.food_types = {
            'apple': {'value': 10, 'grid_id': 2},
            'orange': {'value': 40, 'grid_id': 3},
            'banana': {'value': 100, 'grid_id': 4}
        }
        
        self.reset()

    def reset(self):
        center = self.grid_size // 2
        # Snake starts with length 3, head and two body segments
        self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        self.direction = (1, 0)  # Start facing right (x, y)
        self.score = 0
        
        self.food_positions = {}
        self.spawn_apple()
        self.spawn_orange()
        self.spawn_banana()
        
        self.done = False
        return self.get_observation()
    
    def get_occupied_cells(self):
        """Get all cells occupied by snake and food"""
        occupied = set(self.snake)
        for food_pos in self.food_positions.values():
            if food_pos is not None:
                occupied.add(food_pos)
        return occupied
    
    def get_empty_cells(self):
        """Get all empty cells on the grid"""
        occupied = self.get_occupied_cells()
        empty_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) 
                      if (i, j) not in occupied]
        return empty_cells
    
    def spawn_apple(self):
        """Spawn an apple (10 points)"""
        empty_cells = self.get_empty_cells()
        if empty_cells:
            self.food_positions['apple'] = random.choice(empty_cells)
    
    def spawn_orange(self):
        """Spawn an orange (40 points)"""
        empty_cells = self.get_empty_cells()
        if empty_cells:
            self.food_positions['orange'] = random.choice(empty_cells)
    
    def spawn_banana(self):
        """Spawn a banana (100 points)"""
        empty_cells = self.get_empty_cells()
        if empty_cells:
            self.food_positions['banana'] = random.choice(empty_cells)
    
    def get_observation(self):
        """Create grid representation: 0=empty, 1=snake, 2=apple, 3=orange, 4=banana"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Mark snake positions
        for pos in self.snake:
            grid[pos] = 1
        
        # Mark food positions with their respective IDs
        for food_name, food_pos in self.food_positions.items():
            if food_pos is not None:
                grid[food_pos] = self.food_types[food_name]['grid_id']
        
        return grid
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.done:
            return self.get_observation(), 0, True, {}
        
        # Update direction based on action (0: up, 1: right, 2: down, 3: left)
        # Using (x, y) coordinates where x is horizontal (column) and y is vertical (row)
        direction_map = {
            0: (0, -1),  # up (decrease y)
            1: (1, 0),   # right (increase x)
            2: (0, 1),   # down (increase y)
            3: (-1, 0)   # left (decrease x)
        }
        
        new_direction = direction_map[action]
        # Prevent 180-degree turns
        if (new_direction[0] + self.direction[0], new_direction[1] + self.direction[1]) != (0, 0):
            self.direction = new_direction
        
        # Calculate new head position
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check for collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.done = True
            return self.get_observation(), 0, True, {'score': self.score}
        
        # Check for collision with self
        if new_head in self.snake:
            self.done = True
            return self.get_observation(), 0, True, {'score': self.score}
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if snake ate any food
        reward = 0
        food_eaten = None
        
        for food_name, food_pos in self.food_positions.items():
            if new_head == food_pos:
                food_eaten = food_name
                reward = self.food_types[food_name]['value']
                self.score += reward
                break
        
        if food_eaten:
            # Snake grows, don't remove tail
            # Respawn the eaten food
            if food_eaten == 'apple':
                self.spawn_apple()
            elif food_eaten == 'orange':
                self.spawn_orange()
            elif food_eaten == 'banana':
                self.spawn_banana()
        else:
            # No food eaten, remove tail (snake doesn't grow)
            self.snake.pop()
        
        return self.get_observation(), reward, False, {'score': self.score}
