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

    def get_path_to_food(food_pos):
        path = []
        at_goal = False
        frontier = [(head_pos, [])]
        explored = set()

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance

        while not at_goal:
            if not frontier:
                return []

            # Sort frontier by estimated cost (g + h)

            def estimated_cost(node):
                node_pos, node_path = node
                return len(node_path) + heuristic(node_pos, tuple(closest_food))

            frontier.sort(key=estimated_cost)

            current_pos, current_path = frontier.pop(0) # select the most promising node from the frontier

            # remove duplicates of current_pos from frontier
            frontier = [node for node in frontier if node[0] != current_pos]

            if current_pos == tuple(food_pos): # success: reached food
                path = current_path
                at_goal = True
                break
            
            explored.add(current_pos)

            # Explore neighbors (up, right, down left)
            for move, direction in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                neighbor = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                if (0 <= neighbor[0] < obs.shape[0] and
                    0 <= neighbor[1] < obs.shape[1] and
                    obs[neighbor] != 1 and # not snake body
                    neighbor not in explored):

                    new_path = current_path + [move]
                    frontier.append((neighbor, new_path))

        return path
    
    # find path to closest food, if that returns [], try the next closest food
    sorted_indices = np.argsort(dists)
    #sorted_indices = reversed(sorted_indices) # farthest
    for idx in sorted_indices:
        closest_food = food[idx]
        path = get_path_to_food(closest_food)
        if len(path) > 0:
            return path
    return [] # no path found to any food. we are stuck
    
N_EVAL_EPISODES = 5_000
EVAL_NAME = 'a_star_farthest'

EVERY_MOVE = True

sum_scores = 0
ep_scores = []
ep_steps = []

for eval_ep in range(N_EVAL_EPISODES):
    steps = 0
    
    obs = env.reset()

    path = find_path_with_a_star(obs)

    while True:
        while len(path) > 0:
            action = path.pop(0)
    
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if EVERY_MOVE:
                # for everymove, recompute path to closest food
                path = find_path_with_a_star(obs)
            
            if gui_flag:
                time.sleep(0.05)
                refresh(action, reward, done, info)
        path = find_path_with_a_star(obs)
        if len(path) == 0:
            # stuck - take random actions untl game over
            while not done:
                action = random.randint(0, 3)
                obs, reward, done, info = env.step(action)
                if gui_flag:
                    time.sleep(0.05)
                    refresh(action, reward, done, info)
        if done:
            break

    print('Eval episode %d finished. Score: %d' % (eval_ep+1, env.score))
    sum_scores += env.score
    ep_scores.append(env.score)
    ep_steps.append(steps)

print('Average score over %d eval episodes: %.2f' % (N_EVAL_EPISODES, sum_scores / N_EVAL_EPISODES))
print(f'Max score: {max(ep_scores)}')
print(f'Average steps: {sum(ep_steps)/len(ep_steps)}')

import matplotlib.pyplot as plt

plt.hist(ep_scores, bins=20)
plt.title('Score Distribution over %d Eval Episodes' % N_EVAL_EPISODES)
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.savefig(f'a_star_{EVAL_NAME}_score_distribution.png')

# save ep_scores to a text file
with open(f'a_star_{EVAL_NAME}_scores.txt', 'w') as f:
    for score in ep_scores:
        f.write(f'{score}\n')