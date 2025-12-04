import numpy as np
from typing import Dict, Tuple
from game_engine import UP, DOWN, LEFT, RIGHT, ACTION_NAMES, _slide_and_merge
from heuristics import heuristic_score

ACTIONS = (UP, DOWN, LEFT, RIGHT)


def apply_move_on_board(board: np.ndarray, action: int) -> Tuple[np.ndarray, int, bool]:
    """
    Mirrors Game2048._apply_move + changed flag, but works on a provided board.
    Returns (new_board, merge_reward, changed).
    """
    b = board
    out = np.zeros_like(b)
    reward = 0

    if action == LEFT:
        for i in range(4):
            out[i], g = _slide_and_merge(b[i])
            reward += g
    elif action == RIGHT:
        for i in range(4):
            merged, g = _slide_and_merge(b[i, ::-1])
            out[i] = merged[::-1]
            reward += g
    elif action == UP:
        for j in range(4):
            merged, g = _slide_and_merge(b[:, j])
            out[:, j] = merged
            reward += g
    elif action == DOWN:
        for j in range(4):
            merged, g = _slide_and_merge(b[::-1, j])
            out[:, j] = merged[::-1]
            reward += g
    else:
        raise ValueError(f"Invalid action: {action}")

    changed = not np.array_equal(out, b)
    return out, reward, changed

def legal_moves_for_board(board: np.ndarray):
    legals = []
    for a in ACTIONS:
        _, _, changed = apply_move_on_board(board, a)
        if changed:
            legals.append(a)
    return legals


class ExpectimaxAgent:
    """
    Depth-limited Expectimax with a composite heuristic at leaves.
    - Chance nodes enumerate empty cells and spawn {2,4} with probs {0.9, 0.1}.
    - Optional 'empty_cell_cap' subsamples empties to curb branching when the board is very open.
    - Caching by (board_bytes, is_max_turn, depth) to avoid recomputation.
    """
    def __init__(self, depth: int = 3, empty_cell_cap: int = 6, rng=None, gamma: float = 1.0):
        self.depth = depth
        self.empty_cell_cap = empty_cell_cap
        self.rng = rng or np.random.default_rng()
        self.gamma = gamma
        self.cache: Dict[Tuple[bytes, bool, int], float] = {}

    def select_action(self, game) -> int:
        self.cache.clear()


        board = game.board.copy()
        best_a, best_v = None, -float("inf")

        for a in ACTIONS:
            child, reward, changed = apply_move_on_board(board, a)
            if not changed:
                continue
            v = reward + self.gamma * self._expect_value(child, depth=self.depth-1)
            if v > best_v:
                best_v, best_a = v, a

        return best_a if best_a is not None else UP


    def _max_value(self, board: np.ndarray, depth: int) -> float:
        key = (board.tobytes(), True, depth)
        if key in self.cache:
            return self.cache[key]
        if depth <= 0:
            v = heuristic_score(board)
            self.cache[key] = v
            return v

        legals = legal_moves_for_board(board)
        if not legals:
            v = heuristic_score(board)
            self.cache[key] = v
            return v

        best = -float("inf")
        for a in legals:
            child, reward, _ = apply_move_on_board(board, a)
            v = reward + self.gamma * self._expect_value(child, depth - 1)
            if v > best:
                best = v

        self.cache[key] = best
        return best

    def _expect_value(self, board: np.ndarray, depth: int) -> float:
        key = (board.tobytes(), False, depth)
        if key in self.cache:
            return self.cache[key]
        if depth <= 0:
            v = heuristic_score(board)
            self.cache[key] = v
            return v

        empties = np.argwhere(board == 0)
        if empties.size == 0:
            v = self._max_value(board, depth)
            self.cache[key] = v
            return v

        cells = empties
        if self.empty_cell_cap and len(empties) > self.empty_cell_cap:
            idxs = self.rng.choice(len(empties), size=self.empty_cell_cap, replace=False)
            cells = empties[idxs]

        exp_val = 0.0
        for (i, j) in cells:
            b2 = board.copy(); b2[i, j] = 2
            exp_val += 0.9 * self._max_value(b2, depth)
            b4 = board.copy(); b4[i, j] = 4
            exp_val += 0.1 * self._max_value(b4, depth)

        if len(cells) != len(empties):
            exp_val *= (len(empties) / len(cells))

        self.cache[key] = exp_val
        return exp_val
