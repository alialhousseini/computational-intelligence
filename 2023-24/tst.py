import numpy as np


def calculate_nim_sum(state):
    nim_sum = 0
    for objects in state:
        nim_sum ^= objects
    return nim_sum


def best_move(state):
    nim_sum = calculate_nim_sum(state)

    if nim_sum == 0:
        # No winning move, return a random valid move
        valid_moves = [(pile, objects)
                       for pile, objects in enumerate(state) if objects > 0]
        return valid_moves[np.random.randint(len(valid_moves))]

    # Find a move that makes the nim sum 0
    for pile, objects in enumerate(state):
        target = objects ^ nim_sum
        if target < objects:
            return (pile, objects - target)

    # Fallback, should not reach here in a normal game
    return None


# Example usage
state = np.array([0, 2, 4])  # Example state
pile, objects_to_remove = best_move(state)
print(f"Best Move: Remove {objects_to_remove} objects from pile {pile + 1}")
