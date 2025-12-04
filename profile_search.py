import numpy as np
from typing import Dict, Tuple, Optional
from game_engine import UP, DOWN, LEFT, RIGHT, _slide_and_merge
from heuristics import heuristic_score
from orderings import order_moves
from search_utils import Timer, SearchStats

ACTIONS = (UP, DOWN, LEFT, RIGHT)

def apply_move_on_board(board: np.ndarray, action: int):
    b = board
    out = np.zeros_like(b)
    reward = 0
    if action == LEFT:
        for i in range(4):
            out[i], g = _slide_and_merge(b[i]); reward += g
    elif action == RIGHT:
        for i in range(4):
            merged, g = _slide_and_merge(b[i, ::-1])
            out[i] = merged[::-1]; reward += g
    elif action == UP:
        for j in range(4):
            merged, g = _slide_and_merge(b[:, j])
            out[:, j] = merged; reward += g
    elif action == DOWN:
        for j in range(4):
            merged, g = _slide_and_merge(b[::-1, j])
            out[:, j] = merged[::-1]; reward += g
    else:
        raise ValueError("bad action")
    changed = not np.array_equal(out, b)
    return out, reward, changed

def legal_moves_for_board(board: np.ndarray):
    legals = []
    for a in ACTIONS:
        _, _, changed = apply_move_on_board(board, a)
        if changed: legals.append(a)
    return legals

class ExpectimaxTimeControlled:
    """
    Iterative-deepening expectimax under per-move time budget.
    - timer_budget_sec: per-move time (e.g., 0.05 = 50ms)
    - empty_cell_cap: chance-node subsampling to curb branching
    """
    def __init__(self, timer_budget_sec: float = 0.05, empty_cell_cap: int = 8, gamma: float = 1.0, rng=None):
        self.t_budget = float(timer_budget_sec)
        self.empty_cell_cap = int(empty_cell_cap)
        self.gamma = float(gamma)
        self.rng = rng or np.random.default_rng()
        self.stats = SearchStats()

    def select_action(self, game) -> int:
        board = game.board.copy()
        timer = Timer(self.t_budget); timer.start_now()
        self.stats = SearchStats()

        legals = legal_moves_for_board(board)
        if not legals:
            return UP

        best_move, best_value, best_depth = legals[0], -float("inf"), 0
        depth = 1
        while True:
            if timer.expired(): break
            # transposition cache per iteration to avoid mixing depths
            cache: Dict[Tuple[bytes, bool, int], float] = {}
            # Move ordering
            ordered = order_moves(board, apply_move_on_board)
            mv, val = self._search_root(board, ordered, depth, cache, timer)
            if timer.expired():
                break
            if val is not None:
                best_move, best_value, best_depth = mv, val, depth
            depth += 1

        # Optionally: print or store self.stats
        # print(f"[EX-TC] depth={best_depth} nodes={self.stats.nodes} tt_hits={self.stats.tt_hits}")
        return best_move

    # ----- search internals -----

    def _search_root(self, board, moves, depth, cache, timer):
        best_move, best_val = None, -float("inf")
        for a in moves:
            if timer.expired(): break
            child, reward, changed = apply_move_on_board(board, a)
            if not changed: continue
            v = reward + self.gamma * self._expect(child, depth-1, cache, timer)
            if v > best_val:
                best_val, best_move = v, a
        return best_move, (None if timer.expired() else best_val)

    def _max(self, board, depth, cache, timer):
        self.stats.bump(depth)
        if depth <= 0 or timer.expired():
            return heuristic_score(board)
        key = (board.tobytes(), True, depth)
        if key in cache:
            self.stats.tt_hits += 1
            return cache[key]
        moves = order_moves(board, apply_move_on_board)
        if not moves:
            v = heuristic_score(board); cache[key] = v; self.stats.tt_puts += 1; return v
        best = -float("inf")
        for a in moves:
            if timer.expired(): break
            child, reward, _ = apply_move_on_board(board, a)
            v = reward + self.gamma * self._expect(child, depth-1, cache, timer)
            if v > best: best = v
        cache[key] = best; self.stats.tt_puts += 1
        return best

    def _expect(self, board, depth, cache, timer):
        self.stats.bump(depth)
        if depth <= 0 or timer.expired():
            return heuristic_score(board)
        key = (board.tobytes(), False, depth)
        if key in cache:
            self.stats.tt_hits += 1
            return cache[key]
        empties = np.argwhere(board == 0)
        if empties.size == 0:
            v = self._max(board, depth, cache, timer)
            cache[key] = v; self.stats.tt_puts += 1; return v

        cells = empties
        if self.empty_cell_cap and len(empties) > self.empty_cell_cap:
            idxs = self.rng.choice(len(empties), size=self.empty_cell_cap, replace=False)
            cells = empties[idxs]

        exp_val = 0.0
        for (i, j) in cells:
            if timer.expired(): break
            b2 = board.copy(); b2[i, j] = 2
            exp_val += 0.9 * self._max(b2, depth, cache, timer)
            b4 = board.copy(); b4[i, j] = 4
            exp_val += 0.1 * self._max(b4, depth, cache, timer)

        if len(cells) != len(empties) and len(cells) > 0:
            exp_val *= (len(empties) / len(cells))

        cache[key] = exp_val; self.stats.tt_puts += 1
        return exp_val
