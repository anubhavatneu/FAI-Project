import numpy as np
from game_engine import UP, DOWN, LEFT, RIGHT
from heuristics import heuristic_score

def order_moves(board: np.ndarray, simulate_move_fn):
    scored = []
    for a in (UP, DOWN, LEFT, RIGHT):
        child, reward, changed = simulate_move_fn(board, a)
        if not changed:
            continue
        scored.append((a, reward + 0.2 * heuristic_score(child)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [a for (a, _) in scored]
