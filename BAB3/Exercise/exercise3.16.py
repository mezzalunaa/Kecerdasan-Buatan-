# Exercise 3.16: Q-learning with an eight-state routing diagram
import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(os.path.dirname(current_dir), 'Example')
if example_dir not in sys.path:
    sys.path.append(example_dir)

from q_utils import (
    available_actions,
    best_path,
    create_reward_matrix,
    sample_next_action,
    update_q,
)

points_list = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
]
goal = 7
matrix_size = 8
rewards = create_reward_matrix(matrix_size, points_list, goal)
q_values = np.zeros((matrix_size, matrix_size))
gamma = 0.8
rng = np.random.default_rng(0)

for _ in range(1000):
    current_state = int(rng.integers(matrix_size))
    actions = available_actions(rewards, current_state)
    action = sample_next_action(actions, rng)
    update_q(rewards, q_values, current_state, action, gamma)

print('Most efficient path:', best_path(q_values, 0, goal))
print('Normalized Q matrix:')
print(np.round(q_values / np.max(q_values) * 100, 2))