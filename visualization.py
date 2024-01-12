# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:49:26 2024

@author: mark.pijnenburg
"""
import matplotlib.pyplot as plt
import numpy as np
from gridworld import convert_to_coordinates, convert_to_state, take_action

def visualize_policy(q_values: np.ndarray, grid_size: tuple, pit_locations: list):
    """
    Visualize the learned policy based on the Q-values.

    Parameters:
    - q_values (np.ndarray): Learned Q-values matrix.
    - grid_size (tuple): Size of the gridworld (rows, columns).
    - pit_locations (list): List of integers giving the locations of the pits.
    """
    num_states = grid_size[0] * grid_size[1]
    actions = ['up', 'down', 'left', 'right']

    # Initialize the gridworld
    grid = np.zeros(grid_size)
    
    # Mark pit locations
    for pit in pit_locations:
        row, col = convert_to_coordinates(pit, grid_size)
        grid[row, col] = -1

    # Initialize starting state
    state = 0

    # Visualize the path
    while state != num_states - 1 and state not in pit_locations:
        row, col = convert_to_coordinates(state, grid_size)
        grid[row, col] = 1  # Mark the path
        action = actions[np.argmax(q_values[state, :])]

        state = take_action(state, action, grid_size)

    # Mark the goal state
    row, col = convert_to_coordinates(state, grid_size)
    grid[row, col] = 1
    
    # Plot the gridworld
    plt.imshow(grid, cmap='viridis', origin='upper', interpolation='none')
    plt.title('Learned Policy Visualization')
    plt.colorbar(label='Path (1), Pit (-1)')
    plt.show()

