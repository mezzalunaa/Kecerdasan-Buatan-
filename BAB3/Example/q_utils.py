"""Utilities for small tabular Q-learning routing examples."""
import numpy as np


def create_reward_matrix(matrix_size, points_list, goal):
    rewards = np.full((matrix_size, matrix_size), -1.0)
    for start, end in points_list:
        rewards[start, end] = 100.0 if end == goal else 0.0
        rewards[end, start] = 100.0 if start == goal else 0.0
    rewards[goal, goal] = 100.0
    return rewards


def available_actions(rewards, state):
    return np.flatnonzero(rewards[state] >= 0)


def sample_next_action(actions, rng):
    return int(rng.choice(actions))


def update_q(rewards, q_values, state, action, gamma):
    next_actions = available_actions(rewards, action)
    max_next_value = np.max(q_values[action, next_actions])
    q_values[state, action] = rewards[state, action] + gamma * max_next_value
    return q_values[state, action]


def best_path(q_values, start, goal):
    path = [start]
    state = start
    while state != goal and len(path) <= q_values.shape[0]:
        action = int(np.argmax(q_values[state]))
        if q_values[state, action] <= 0 or action in path:
            break
        path.append(action)
        state = action
    return path
