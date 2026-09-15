# Example 3.22 Q-learning routing
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from q_utils import (
    available_actions,
    best_path,
    create_reward_matrix,
    sample_next_action,
    update_q,
)

points_list = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (3, 6)]
goal = 6
matrix_size = 7

rewards = create_reward_matrix(matrix_size, points_list, goal)
q_values = np.zeros((matrix_size, matrix_size))
gamma = 0.8
rng = np.random.default_rng(0)
scores = []

for _ in range(700):
    current_state = int(rng.integers(matrix_size))
    actions = available_actions(rewards, current_state)
    action = sample_next_action(actions, rng)
    update_q(rewards, q_values, current_state, action, gamma)
    scores.append(np.sum(q_values / np.max(q_values) * 100) if np.max(q_values) else 0)

print('Trained Q matrix:')
print(np.round(q_values / np.max(q_values) * 100, 2))

print('\nMost efficient path:')
print(best_path(q_values, 0, goal))

plt.plot(scores)
plt.xlabel('Training iteration')
plt.ylabel('Q score')
plt.title('Q-learning Training')
plt.grid(True)
plt.show()