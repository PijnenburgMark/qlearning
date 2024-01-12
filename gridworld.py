# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:48:43 2024

@author: mark.pijnenburg
"""

def convert_to_coordinates(state: int, grid_size: tuple) -> tuple:
    return divmod(state, grid_size[1])

def convert_to_state(row: int, col: int, grid_size: tuple) -> int:
    return row * grid_size[1] + col

def take_action(state: int, action: str, grid_size: tuple) -> int:
    row, col = convert_to_coordinates(state, grid_size)
    next_row, next_col = row, col  # Default to the current position

    if action == 'up' and row > 0:
        next_row -= 1
    elif action == 'down' and row < grid_size[0] - 1:
        next_row += 1
    elif action == 'left' and col > 0:
        next_col -= 1
    elif action == 'right' and col < grid_size[1] - 1:
        next_col += 1

    return convert_to_state(next_row, next_col, grid_size)
