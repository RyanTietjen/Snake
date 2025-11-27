"""
File usage:
- Update hash function, num_episodes, and decay_rate.
- To train: python Q_learning.py train
- To evaluate with GUI: python Q_learning.py gui
- To evaluate without GUI: python Q_learning.py 
"""
import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from snake_gui import *

train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

setup(GUI=gui_flag)
env = game # Gym environment already initialized within snake_gui.py

"""
whole_board
hash_fruit_directions
hash_fruit_directions_with_snake_direction
hash_fruit_directions_with_snake_direction_3_3_surroundings
hash_fruit_directions_with_snake_direction_4_4_surroundings
"""
ACTIVE_HASH_FUNCTION = 'hash_fruit_directions_with_snake_direction' 

num_episodes = 1000
decay_rate = 0.999

# ============================================================================
# HASH FUNCTIONS
# ============================================================================
def hash_whole_board_all_info():
    """
    Hashes all info of the board, including:
    - Snake head, direction its looking, and body positions (in order)
    - Positions of all three fruits (apple, orange, banana)
    
    # of unique states is extremely large
    
    Example:
    5_5,4_5,3_5|10|8_6,6_2,1_3
    Snake is in starting position, looking right, with apple at (8,6), banana at (6,2), orange at (1,3)
    """
    snake_str = ','.join(f"{x}_{y}" for x, y in env.snake)
    dir_str = f"{env.direction[0]}{env.direction[1]}"
    
    # Sort for consistentcy
    food_items = sorted(env.food_positions.items())
    apple_pos = str(food_items[0][1]).replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
    banana_pos = str(food_items[1][1]).replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
    orange_pos = str(food_items[2][1]).replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
    
    return f"{snake_str}|{dir_str}|{apple_pos},{banana_pos},{orange_pos}"

def hash_fruit_directions():
    """
    Hashes the relative directions of the fruits from the snake's head
    in the order apple, banana, orange.

    4^3 = 64 unique states

    Example:
    NSE 
    apple is north, banana is south, orange is east of the snake head.
    """
    return get_fruit_directions(env.snake[0], env.food_positions)

def hash_fruit_directions_with_snake_direction():
    """
    Hashes only the relative directions of the fruits from the snake's head
    in the order apple, banana, orange.
    
    4^3 * 4 = 256 unique states

    Example:
    NSE|10 
    apple is north, banana is south, orange is east, with snake facing direction (1,0) (East).
    """
    dir_str = f"{env.direction[0]}{env.direction[1]}"
    food_str = get_fruit_directions(env.snake[0], env.food_positions)
    return f"{food_str}|{dir_str}"

def hash_fruit_directions_with_snake_direction_3_3_surroundings():
    """
    Hashes the relative directions of the fruits, snake direction, and 3x3 surrounding tiles.
    
    4^3 * 4 * 2^8 = 65536 unique states
    
    Example:
    EEW|10|00010000
    - apple is east, banana is east, orange is west
    - snake facing direction (1,0) (East)
    - surrounding tiles (N, NE, E, SE, S, SW, W, NW) with 1=body/wall, 0=empty
    """
    head_x, head_y = env.snake[0]
    dir_str = f"{env.direction[0]}{env.direction[1]}"
    food_str = get_fruit_directions(env.snake[0], env.food_positions)
    
    surrounding_offsets = []
    for y in range(-1, 2):
        for x in range(-1, 2):
            if x != 0 or y != 0: 
                surrounding_offsets.append((x, y))
    
    surrounding = ""
    for x, y in surrounding_offsets:
        adj_pos = (head_x + x, head_y + y)
        if (adj_pos[0] < 0 or adj_pos[0] >= env.grid_size or 
            adj_pos[1] < 0 or adj_pos[1] >= env.grid_size or 
            adj_pos in env.snake[1:]):
            surrounding += "1"
        else:
            surrounding += "0"
    
    return f"{food_str}|{dir_str}|{surrounding}"

def hash_fruit_directions_with_snake_direction_4_4_surroundings():
    """
    Hashes the relative directions of the fruits, snake direction, and 4x4 surrounding tiles.

    4^3 * 4 * 2^15 = 8,388,608 unique states
    
    Example:
    NNS|10|000000001100000 means:
    - apple is north, banana is north, orange is south
    - snake facing direction (1,0) (East)
    - surrounding tiles in 4x4 grid with 1=body/wall, 0=empty  
    """
    head_x, head_y = env.snake[0]
    dir_str = f"{env.direction[0]}{env.direction[1]}"
    food_str = get_fruit_directions(env.snake[0], env.food_positions)
    
    surrounding_offsets = []
    for dy in range(-2, 2):
        for dx in range(-2, 2):
            if dx != 0 or dy != 0: 
                surrounding_offsets.append((dx, dy))
    
    surrounding = ""
    for x, y in surrounding_offsets:
        adj_pos = (head_x + x, head_y + y)
        if (adj_pos[0] < 0 or adj_pos[0] >= env.grid_size or 
            adj_pos[1] < 0 or adj_pos[1] >= env.grid_size or 
            adj_pos in env.snake[1:]):
            surrounding += "1"
        else:
            surrounding += "0"
    
    return f"{food_str}|{dir_str}|{surrounding}"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_fruit_directions(head_pos, food_positions):
    """
    This helper function was mostly generated by GitHub Copilot.
    Prompt: Create a helper function that calculates the cardinal direction to each fruit,
    prioritizing the closest axis distance. For example, if the banana is up 2 and right 5,
    the direction should be N.
    
    Calculate the direction to each fruit based on closest axis distance.
    
    Args:
        head_pos: Tuple of (x, y) for snake head position
        food_positions: Dictionary of {fruit_type: (x, y)} positions
    
    Returns:
        String of concatenated direction letters in alphabetical order by fruit type
    """
    head_x, head_y = head_pos
    food_items = sorted(food_positions.items())
    fruit_directions = []
    
    for ftype, (fruit_x, fruit_y) in food_items:
        # Calculate distances in each direction
        dist_vertical = abs(fruit_y - head_y)
        dist_horizontal = abs(fruit_x - head_x)
        
        # Prioritize the direction with smaller distance
        if dist_vertical < dist_horizontal:
            # Vertical direction is closer
            if fruit_y < head_y:
                direction = 'N'
            elif fruit_y > head_y:
                direction = 'S'
            else:  # Same y coordinate
                if fruit_x > head_x:
                    direction = 'E'
                else:
                    direction = 'W'
        elif dist_horizontal < dist_vertical:
            # Horizontal direction is closer
            if fruit_x > head_x:
                direction = 'E'
            elif fruit_x < head_x:
                direction = 'W'
            else:  # Same x coordinate
                if fruit_y < head_y:
                    direction = 'N'
                else:
                    direction = 'S'
        else:
            # Equal distance - prioritize N/S over E/W
            if fruit_y < head_y:
                direction = 'N'
            elif fruit_y > head_y:
                direction = 'S'
            elif fruit_x > head_x:
                direction = 'E'
            elif fruit_x < head_x:
                direction = 'W'
            else:
                direction = 'X'
        
        fruit_directions.append(direction)
    
    return ''.join(fruit_directions)

HASH_FUNCTIONS = {
    'whole_board': hash_whole_board_all_info,
    'hash_fruit_directions': hash_fruit_directions,
    'hash_fruit_directions_with_snake_direction': hash_fruit_directions_with_snake_direction,
    'hash_fruit_directions_with_snake_direction_3_3_surroundings': hash_fruit_directions_with_snake_direction_3_3_surroundings,
    'hash_fruit_directions_with_snake_direction_4_4_surroundings': hash_fruit_directions_with_snake_direction_4_4_surroundings
}

hash_state = HASH_FUNCTIONS[ACTIVE_HASH_FUNCTION]

print(f"Using hash function: {ACTIVE_HASH_FUNCTION}")
print(f"Example hash: {hash_state()}") 

# ============================================================================
# Q Learning
# ============================================================================
def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}
    num_updates = {}  # η = 1/(1 + number of updates to Q_opt(s,a))
                      # Q_opt === N(s, a)
    episode_rewards = []  
 
    for _ in tqdm(range(num_episodes)):
        obs = env.reset()
        total_episode_reward = 0
  
        while True:
            state = hash_state()
            """
            For a state that does not already have a corresponding entry in the Q-table dict, you
            should add an entry with the initial Q-value estimates for all actions from that state set to 0. 
            """
            if state not in Q_table:
                Q_table[state] = np.zeros(env.action_space.n)
                num_updates[state] = np.zeros(env.action_space.n)
    
            if np.random.rand() < epsilon:
                # random(a∈A); with probability P = ε
                action = env.action_space.sample()
            else:
                # arg max_a∈A Qopt(s,a); with probability P = (1 − ε)
                action = np.argmax(Q_table[state])

            obs, reward, done, info = env.step(action)
            
            reward = reward - 2

            if done:
                reward = reward - 100000
            
            total_episode_reward += reward

            # Update Q-values
            if done:
                V_next = 0
            else:
                next_state = hash_state()
                # Need to initialize values for new states, crashes otherwise
                if next_state not in Q_table:
                    Q_table[next_state] = np.zeros(env.action_space.n)
                    num_updates[next_state] = np.zeros(env.action_space.n)
                # V̂_old_opt(s') = max_a' Q̂_old_opt(s',a')
                V_next = np.max(Q_table[next_state])

            # Q̂_new_opt(s,a) = (1-η)Q̂_old_opt(s,a) + η[R(s,a,s') + γV̂_old_opt(s')]
            eta = 1.0 / (1 + num_updates[state][action])
            Q_old = Q_table[state][action]
            Q_table[state][action] = (1 - eta) * Q_old + eta * (reward + gamma * V_next)
            num_updates[state][action] += 1
   
            if done:
                break

        episode_rewards.append(total_episode_reward)
        epsilon *= decay_rate

    return Q_table

# ============================================================================
# Training & Evaluation
# ============================================================================
if train_flag:
    Q_table = Q_learning(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

    # Save the Q-table dict to a file
    filename = f'Q_table_{ACTIVE_HASH_FUNCTION}_{num_episodes}_{decay_rate}.pickle'
    print("Saving file:", filename)
    with open(filename, 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Q-table saved.")
    
        
def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)

if not train_flag:
    # Load the Q-table dict from a file
    filename = f'Q_table_{ACTIVE_HASH_FUNCTION}_{num_episodes}_{decay_rate}.pickle'
    print(f"Loading Q-table from {filename}...")
    start_load = time.time()
    with open(filename, 'rb') as handle:
        Q_table = pickle.load(handle)
    load_time = time.time() - start_load
    
    print(f"Q-table loaded successfully in {load_time:.2f} seconds!")
    
    num_eval_episodes = 10000
    episode_rewards = []
    moves_made_per_episode = []
    unseen_states = set()
    q_table_actions = 0
    random_actions = 0

    start = time.time()
    
    for episode in tqdm(range(num_eval_episodes), desc="Evaluating"):
        obs = env.reset()
        total_reward = 0
        moves_made = 0

        while True:
            state = hash_state()
            if state not in Q_table:
                action = env.action_space.sample()
                unseen_states.add(state)
                random_actions += 1
            else:
                q_table_actions += 1
                # Use softmax action selection for evaluation
                action_probs = softmax(Q_table[state], temp=0.1)
                action = np.random.choice(np.arange(env.action_space.n), p=action_probs)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            moves_made += 1

            if gui_flag:
                refresh(obs, reward, done, info, delay=0.05)

            if done:
                episode_rewards.append(total_reward)
                moves_made_per_episode.append(moves_made)
                break
            
    avg_reward = np.mean(episode_rewards)
    eval_time = time.time() - start

    print(f"Number of unique states in Q-table: {len(Q_table)}")
    print(f"Average reward over 10,000 evaluation episodes: {avg_reward:.4f}")
    print(f"Average episode length (number of actions): {np.mean(moves_made_per_episode):.4f}")
    print(f"Total time for 10,000 evaluation episodes: {eval_time:.4f} seconds")
    print(f"Number of unique states encountered during evaluation not in Q-table: {len(unseen_states)}")
    print(f"Percentage of actions taken using Q-table: {q_table_actions / (q_table_actions + random_actions):.4f}")
    print(f"Percentage of random actions due to missing Q-values: {random_actions / (q_table_actions + random_actions):.4f}")