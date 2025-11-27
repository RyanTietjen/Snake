"""
A* Pathfinding Algorithm Implementation for Snake Game
File usage:
- To evaluate with GUI: python a_star.py gui
- To evaluate without GUI: python a_star.py 
"""

import sys
from snake_gui import *
import numpy as np

gui_flag = 'gui' in sys.argv

print(f'Using gui: {gui_flag}')
setup(GUI=gui_flag)

env = game
obs = env.reset()

def find_path_with_a_star(obs):
    #0=empty, 1=snake_body, 2=apple, 3=orange, 4=banana, 5=snake_head
    
    # locate snake head
    head_pos = np.where(obs == 5)
    assert(len(head_pos[0]) == 1 and len(head_pos[1]) == 1) # let's not get ahead of ourselves :)
    head_pos = head_pos[0][0], head_pos[1][0]

    apples = np.array(np.where(obs == 2)).T
    oranges = np.array(np.where(obs == 3)).T
    bananas = np.array(np.where(obs == 4)).T

    food = np.concatenate((apples, oranges, bananas), axis=0)

    # get closest food
    if len(food) == 0:
        print("No food found!")
        return []
    dists = np.linalg.norm(food - head_pos, axis=1)
    closest_food = food[np.argmin(dists)]

    path = []
    at_goal = False
    frontier = [(head_pos, [])]
    explored = set()
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance
    
    while not at_goal:
        if not frontier:
            print("No path found!")
            return []
        
        # Sort frontier by estimated cost (g + h)

        def estimated_cost(node):
            node_pos, node_path = node
            return len(node_path) + heuristic(node_pos, tuple(closest_food))

        frontier.sort(key=estimated_cost)

        current_pos, current_path = frontier.pop(0) # select the most promising node from the frontier
        
        if current_pos == tuple(closest_food): # success: reached food
            path = current_path
            at_goal = True
            break
        
        explored.add(current_pos)
        
        # Explore neighbors (up, down, left, right)
        for move, direction in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
            neighbor = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            if (0 <= neighbor[0] < obs.shape[0] and
                0 <= neighbor[1] < obs.shape[1] and
                obs[neighbor] != 1 and # not snake body
                neighbor not in explored):
                
                new_path = current_path + [move]
                frontier.append((neighbor, new_path))
    
    return path

find_path_with_a_star(obs)

#time.sleep(1)
#
#obs, reward, done, info = env.step(1)
## obs is 0=empty, 1=snake_body, 2=apple, 3=orange, 4=banana, 5=snake_head
#
#refresh(1, reward, done, info)
#time.sleep(1)
#
#obs, reward, done, info = env.step(2)
#refresh(2, reward, done, info)
#time.sleep(1)

