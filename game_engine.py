import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

def _slide_and_merge(row: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Slides non-zero tiles to the left and merges equal adjacent tiles once.
    Returns the new row and the score gained by merges in this row.
    """
    non_zero = row[row != 0]
    if non_zero.size == 0:
        return np.zeros_like(row), 0

    merged = []
    score_gain = 0
    i = 0
    while i < len(non_zero):
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
            new_val = non_zero[i] * 2
            merged.append(new_val)
            score_gain += new_val
            i += 2
        else:
            merged.append(non_zero[i])
            i += 1

    merged = np.array(merged, dtype=row.dtype)
    out = np.zeros_like(row)
    out[:merged.size] = merged
    return out, score_gain


@dataclass
class Game2048:
    seed: Optional[int] = None
    board: np.ndarray = field(default_factory=lambda: np.zeros((4,4), dtype=np.int64))
    score: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if not self.board.any():
            self.reset(self.seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the game to a starting state with two random tiles."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.board[:] = 0
        self.score = 0
        self._spawn_tile()
        self._spawn_tile()
        return self.board.copy()

    def clone(self) -> "Game2048":
        g = Game2048(seed=self.seed)
        g.board = self.board.copy()
        g.score = self.score
        g.rng = np.random.default_rng()
        return g

    def _spawn_tile(self) -> bool:
        """Spawn a 2 (90%) or 4 (10%) in a random empty cell."""
        empties = np.argwhere(self.board == 0)
        if empties.size == 0:
            return False
        idx = self.rng.integers(0, len(empties))
        i, j = empties[idx]
        self.board[i, j] = 4 if self.rng.random() < 0.1 else 2
        return True

    def legal_moves(self) -> List[int]:
        """Return list of legal moves (those that change the board)."""
        legals = []
        for a in (UP, DOWN, LEFT, RIGHT):
            _, _, changed = self._peek(a)
            if changed:
                legals.append(a)
        return legals

    def _apply_move(self, action: int) -> Tuple[np.ndarray, int]:
        """Apply action to board (without spawning tile)."""
        b = self.board
        out = np.zeros_like(b)
        score_gain_total = 0

        if action == LEFT:
            for i in range(4):
                out[i], gain = _slide_and_merge(b[i])
                score_gain_total += gain
        elif action == RIGHT:
            for i in range(4):
                row = b[i, ::-1]
                merged, gain = _slide_and_merge(row)
                out[i] = merged[::-1]
                score_gain_total += gain
        elif action == UP:
            for j in range(4):
                col = b[:, j]
                merged, gain = _slide_and_merge(col)
                out[:, j] = merged
                score_gain_total += gain
        elif action == DOWN:
            for j in range(4):
                col = b[::-1, j]
                merged, gain = _slide_and_merge(col)
                out[:, j] = merged[::-1]
                score_gain_total += gain
        else:
            raise ValueError(f"Invalid action: {action}")

        return out, score_gain_total

    def _peek(self, action: int) -> Tuple[np.ndarray, int, bool]:
        """Return (candidate_board, score_gain, changed_flag) without mutating game state."""
        new_board, gain = self._apply_move(action)
        changed = not np.array_equal(new_board, self.board)
        return new_board, gain, changed

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, Dict]:
        """Take a step: if move changes the board, apply it, spawn a tile, update score."""
        new_board, gain, changed = self._peek(action)
        if not changed:
            return self.board.copy(), 0, self.is_game_over(), {"changed": False, "invalid": True}

        self.board = new_board
        self.score += gain
        self._spawn_tile()
        done = self.is_game_over()
        return self.board.copy(), gain, done, {"changed": True, "invalid": False}

    def is_game_over(self) -> bool:
        if np.any(self.board == 0):
            return False
        for a in (UP, DOWN, LEFT, RIGHT):
            _, _, changed = self._peek(a)
            if changed:
                return False
        return True

    def max_tile(self) -> int:
        return int(self.board.max())

    def render(self) -> None:
        print("+------+------+------+------+")
        for i in range(4):
            row = "|".join(f"{v:>6}" if v != 0 else "      " for v in self.board[i])
            print(f"|{row}|")
            print("+------+------+------+------+")
        print(f"Score: {self.score}")
