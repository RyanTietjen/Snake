
import sys
from snake_gui import *
import numpy as np

gui_flag = 'gui' in sys.argv

print(f'Using gui: {gui_flag}')
setup(GUI=gui_flag)

direction_map = {
    'up': 0,
    'right': 1,
    'down': 2,
    'left': 3
}

env = game
obs = env.reset()

cycle = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [0, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 2],
    [0, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 2],
    [0, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 2],
    [0, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 2],
    [0, 3, 3, 3, 3, 3, 3, 3, 3, 3]
]

n_epiodes = 5000
scores = []
step_counts = []
for i in range(n_epiodes):
    obs = env.reset()
    ep_steps = 0

    while True:
        snake_x, snake_y = env.snake[0]

        action = cycle[snake_y][snake_x]

        obs, reward, done, info = env.step(action)
        ep_steps += 1
        
        if gui_flag:
            #time.sleep(0.01)
            refresh(action, reward, done, info)
        if done:
            print(f'Episode {i}/{n_epiodes} ended with score {env.score}')
            scores.append(env.score)
            step_counts.append(ep_steps)
            break


print(f'Average score over {n_epiodes} episodes: {np.mean(scores)}')
print(f'Max score: {max(scores)}')
print(f'Avg step count: {sum(step_counts)/len(step_counts)}')