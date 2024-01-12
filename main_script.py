# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:47:32 2024

@author: mark.pijnenburg
"""
import numpy as np
from utils import initialize_q_values, choose_action, update_q_values
from gridworld import take_action
from visualization import visualize_policy

def run_q_learning(grid_size: tuple, pit_locations: list,
                   num_episodes: int,
                   learning_rate: float = 0.1,
                   discount_factor: float = 0.9,
                   exploration_prob: float = 0.2) -> np.ndarray:
    num_states = grid_size[0] * grid_size[1]
    actions = ['up', 'down', 'left', 'right']

    q_values = initialize_q_values(num_states, len(actions))

    for _ in range(num_episodes):
        state = 0  # Start at state 0 (corresponding to (0,0) coordinates)

        while state != num_states - 1 and state not in pit_locations:
            action = actions[choose_action(q_values, state, grid_size, 
                                           exploration_prob)]
            next_state = take_action(state, action, grid_size)
            reward = 1 if next_state == num_states - 1 else -1 if next_state in pit_locations else 0

            q_values = update_q_values(q_values, state, actions.index(action),
                                       reward, next_state, learning_rate,
                                       grid_size, discount_factor)

            state = next_state

    return q_values

def main():
    grid_size = (8, 8)
    pit_locations = [2,5,8,16, 34,35]
    num_episodes = 1000
    learned_q_values = run_q_learning(grid_size, pit_locations, num_episodes)

    # Print the learned Q-values
    print("Learned Q-values:")
    print(learned_q_values)

    # Visualize the learned policy
    visualize_policy(learned_q_values, grid_size, pit_locations)

if __name__ == "__main__":
    main()
