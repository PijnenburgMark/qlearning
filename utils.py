import numpy as np
from gridworld import convert_to_coordinates

def initialize_q_values(num_states: int, num_actions: int) -> np.ndarray:
    return np.zeros((num_states, num_actions))

def choose_action(q_values: np.ndarray, state: int, 
                  grid_size: tuple, exploration_prob: float) -> int:
    """
    Choose an action using an epsilon-greedy strategy.

    Parameters:
    - q_values (np.ndarray): Q-values matrix.
    - state (int): The current state.
    - grid_size (tuple): size of the board.
    - exploration_prob (float): Probability of exploration.
    

    Returns:
    - int: Chosen action.
    """
    valid_actions = get_valid_actions(state, grid_size)  # Get valid actions for the current state

    if np.random.rand() < exploration_prob:
        return np.random.choice(valid_actions)  # Explore
    else:
        max_q_value = np.max(q_values[state, valid_actions])
        max_indices = np.where(q_values[state, valid_actions] == max_q_value)[0]
        return valid_actions[np.random.choice(max_indices)]  # Randomly choose among max Q-values

def get_valid_actions(state: int, grid_size: tuple) -> list:
    """
    Get valid actions for a given state.

    Parameters:
    - state (int): The current state.
    - grid_size (tuple): size of the board.

    Returns:
    - list: List of valid actions.
    """
    row, col = convert_to_coordinates(state, grid_size)
    valid_actions = []

    if row > 0:  # Can move up
        valid_actions.append(0)  # Up
    if row < grid_size[0] - 1:  # Can move down
        valid_actions.append(1)  # Down
    if col > 0:  # Can move left
        valid_actions.append(2)  # Left
    if col < grid_size[1] - 1:  # Can move right
        valid_actions.append(3)  # Right

    return valid_actions

def update_q_values(q_values: np.ndarray, state: int, action: int,
                    reward: float, next_state: int,
                    learning_rate: float, grid_size: tuple, 
                    discount_factor: float) -> np.ndarray:
    """
    Update Q-values based on the Q-learning update rule.

    Parameters:
    - q_values (np.ndarray): Q-values matrix.
    - state (int): Current state.
    - action (int): Chosen action.
    - reward (float): Received reward.
    - next_state (int): Next state.
    - learning_rate (float): Learning rate.
    - discount_factor (float): Discount factor for future rewards.
    - grid_size (tuple): size of the board

    Returns:
    - np.ndarray: Updated Q-values matrix.
    """
    best_next_action = choose_action(q_values, next_state, grid_size, exploration_prob=0)
    q_values[state, action] += learning_rate * (
        reward + discount_factor * q_values[next_state, best_next_action]
        - q_values[state, action]
    )
    return q_values
