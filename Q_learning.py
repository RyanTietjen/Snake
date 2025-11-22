import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from snake_gui import *
import matplotlib.pyplot as plt

train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

setup(GUI=gui_flag)
env = game # Gym environment already initialized within snake_gui.py

# ============================================================================
# HASH FUNCTIONS - Add new hash functions here for experimentation
# ============================================================================

def hash_whole_board_all_info():
    """Hash the entire board state for use as a key in Q-table
    
    Captures:
    - Snake head and body positions
    - Direction the snake is facing
    - Positions of all three fruits (apple, orange, banana)
    
    Returns a string representation that can be used as a dictionary key
    """
    # Get snake positions and convert to string
    snake_str = ','.join(f"{x}_{y}" for x, y in env.snake)
    
    # Get direction
    dir_str = f"{env.direction[0]}_{env.direction[1]}"
    
    # Get food positions (sorted for consistency)
    food_items = sorted(env.food_positions.items())
    food_str = ','.join(f"{ftype}:{pos[0]}_{pos[1]}" for ftype, pos in food_items)
    
    # Combine all components into a single string separated by '|'
    state_hash = f"{snake_str}|{dir_str}|{food_str}"
    
    return state_hash

def hash_fruit_directions_with_snake_direction_3_3_surroundings():
    """
    Hashes the relative directions of the fruits, snake direction, and 3x3 surrounding tiles.
    
    Includes:
    - Fruit directions (apple, banana, orange)
    - Snake facing direction
    - Which of the 8 adjacent tiles (including diagonals) contain snake body
    
    For example:
    - 'NSE0110100000' means:
      - apple is north, banana is south, orange is east
      - snake facing direction (0,1) (Right)
      - surrounding tiles (N, NE, E, SE, S, SW, W, NW) with 1=body, 0=empty

    4^3 * 4 * 2^8 = 65536 unique states
    """
    head_x, head_y = env.snake[0]
    dir_str = f"{env.direction[0]}{env.direction[1]}"
    food_str = get_fruit_directions(env.snake[0], env.food_positions)
    
    # Check 8 adjacent tiles for snake body (N, NE, E, SE, S, SW, W, NW)
    adjacent_offsets = [
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # E
        (1, 1),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0),   # W
        (-1, -1),  # NW
    ]
    
    # Convert snake body to set for O(1) lookup (exclude head)
    snake_body_set = set(env.snake[1:])
    
    # Check each adjacent position
    surrounding = []
    for dx, dy in adjacent_offsets:
        adj_pos = (head_x + dx, head_y + dy)
        if adj_pos in snake_body_set:
            surrounding.append('1')
        else:
            surrounding.append('0')
    
    surrounding_str = ''.join(surrounding)
    
    return f"{food_str}{dir_str}{surrounding_str}"

def hash_fruit_directions_with_snake_direction_4_4_surroundings():
    """
    Hashes the relative directions of the fruits, snake direction, and 4x4 surrounding tiles.
    
    Includes:
    - Fruit directions (apple, banana, orange)
    - Snake facing direction
    - Which of the tiles in a 4x4 grid centered on the snake head contain snake body or walls
    
    The 4x4 grid includes:
    - 2 tiles in each direction from the head (excludes the head itself)
    - Total of 15 tiles checked (4x4 = 16, minus the head position = 15)
    - Tiles that are walls (out of bounds) are marked as 1
    
    For example:
    - 'NSE01000100010001000' means:
      - apple is north, banana is south, orange is east
      - snake facing direction (0,1) (Right)
      - surrounding tiles in 4x4 grid with 1=body/wall, 0=empty
    
    4^3 * 4 * 2^15 = 8,388,608 unique states
    """
    head_x, head_y = env.snake[0]
    dir_str = f"{env.direction[0]}{env.direction[1]}"
    food_str = get_fruit_directions(env.snake[0], env.food_positions)
    
    # Check 4x4 grid around snake head (excluding the head itself)
    # Grid goes from -2 to +1 in both x and y directions from head
    surrounding_offsets = []
    for dy in range(-2, 2):
        for dx in range(-2, 2):
            if dx != 0 or dy != 0:  # Exclude the head position itself
                surrounding_offsets.append((dx, dy))
    
    # Convert snake body to set for O(1) lookup (exclude head)
    snake_body_set = set(env.snake[1:])
    
    # Check each position in the 4x4 grid
    surrounding = []
    for dx, dy in surrounding_offsets:
        adj_pos = (head_x + dx, head_y + dy)
        # Check if position is wall (out of bounds) or contains snake body
        if (adj_pos[0] < 0 or adj_pos[0] >= env.grid_size or 
            adj_pos[1] < 0 or adj_pos[1] >= env.grid_size or 
            adj_pos in snake_body_set):
            surrounding.append('1')
        else:
            surrounding.append('0')
    
    surrounding_str = ''.join(surrounding)
    
    return f"{food_str}{dir_str}{surrounding_str}"

def hash_fruit_directions_with_snake_direction():
    """
    Hashes only the relative directions of the fruits from the snake's head
    in the order apple, banana, orange.

    Prioritizes the direction with the smaller distance.

    For example:
    - 'NSE01' means apple is north, banana is south, orange is east of the snake head, with snake facing direction (0,1) (Right).
    - 'WNN0-1' means apple is west, banana is north, orange is north of the snake head, with snake facing direction (0,-1) (Left).

    4^3 * 4 = 256 unique states
    
    """
    head_pos = env.snake[0]
    dir_str = f"{env.direction[0]}{env.direction[1]}"
    food_str = get_fruit_directions(head_pos, env.food_positions)
    return f"{food_str}{dir_str}"

def hash_fruit_directions():
    """
    Hashes the relative directions of the fruits from the snake's head
    in the order apple, banana, orange.

    Also includes the current direction the snake is facing.

    Prioritizes the direction with the smaller distance.

    For example:
    - 'NSE' means apple is north, banana is south, orange is east of the snake head.

    4^3 = 64 unique states
    
    """
    head_pos = env.snake[0]
    return get_fruit_directions(head_pos, env.food_positions)

# ============================================================================
# Helper functions for hashing
# ============================================================================
def get_fruit_directions(head_pos, food_positions):
    """Calculate the direction to each fruit based on closest axis distance.
    
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

# ============================================================================
# HASH FUNCTION REGISTRY
# ============================================================================
# Add your hash functions to this dictionary to easily switch between them
HASH_FUNCTIONS = {
    'whole_board': hash_whole_board_all_info,
    'hash_fruit_directions': hash_fruit_directions,
    'hash_fruit_directions_with_snake_direction': hash_fruit_directions_with_snake_direction,
    'hash_fruit_directions_with_snake_direction_3_3_surroundings': hash_fruit_directions_with_snake_direction_3_3_surroundings,
    'hash_fruit_directions_with_snake_direction_4_4_surroundings': hash_fruit_directions_with_snake_direction_4_4_surroundings
}

# Select which hash function to use (change this to test different approaches)
ACTIVE_HASH_FUNCTION = 'hash_fruit_directions_with_snake_direction_4_4_surroundings' 

hash_state = HASH_FUNCTIONS[ACTIVE_HASH_FUNCTION]

num_episodes = 1000000
decay_rate = 0.999999

print(f"Using hash function: {ACTIVE_HASH_FUNCTION}")
print(f"Example hash: {hash_state()}") 


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
        
    episode_rewards_array = np.array(episode_rewards)
    cumulative_sum = np.cumsum(episode_rewards_array)
    episode_numbers = np.arange(1, len(episode_rewards) + 1)
    running_avg = cumulative_sum / episode_numbers

    # plt.figure(figsize=(12, 6))
    # plt.plot(running_avg, linewidth=2)
    
    # plt.xlabel('Episode', fontsize=12, fontweight='bold')
    # plt.ylabel('Average Reward', fontsize=12, fontweight='bold')
    # plt.ylim(bottom=None, top=11000)
    # plt.title(f'Cumulative Running Average Reward per Episode During Training\nHash: {ACTIVE_HASH_FUNCTION}, Episodes: {num_episodes}, Decay: {decay_rate}', 
    #           fontsize=14, fontweight='bold', pad=20)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    
    # training_plot_filename = f'training_rewards_{ACTIVE_HASH_FUNCTION}_{num_episodes}_{decay_rate}.png'
    # plt.savefig(training_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')

    return Q_table

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
        
    
    # Print evaluation statistics

    avg_reward = np.mean(episode_rewards)
    eval_time = time.time() - start

    print(f"Number of unique states in Q-table: {len(Q_table)}")
    print(f"Average reward over 10,000 evaluation episodes: {avg_reward:.4f}")
    print(f"Average episode length (number of actions): {np.mean(moves_made_per_episode):.4f}")
    print(f"Total time for 10,000 evaluation episodes: {eval_time:.4f} seconds")
    print(f"Number of unique states encountered during evaluation not in Q-table: {len(unseen_states)}")
    print(f"Percentage of actions taken using Q-table: {q_table_actions / (q_table_actions + random_actions):.4f}")
    print(f"Percentage of random actions due to missing Q-values: {random_actions / (q_table_actions + random_actions):.4f}")