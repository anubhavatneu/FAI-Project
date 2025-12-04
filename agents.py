import numpy as np
from game_engine import Game2048, UP, DOWN, LEFT, RIGHT

class RandomAgent:
    """Chooses a random legal move each turn."""
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def select_action(self, game: Game2048) -> int:
        legals = game.legal_moves()
        if not legals:
            return LEFT
        return int(self.rng.choice(legals))


class GreedyImmediateAgent:
    """Greedy agent that picks move with highest immediate merge reward."""
    def _empty_cells(self, board: np.ndarray) -> int:
        return int(np.sum(board == 0))

    def _corner_bonus(self, board: np.ndarray) -> int:
        maxv = board.max()
        corners = [board[0,0], board[0,3], board[3,0], board[3,3]]
        return 1 if maxv in corners else 0

    def select_action(self, game: Game2048) -> int:
        best = None
        best_tuple = None
        for a in (UP, DOWN, LEFT, RIGHT):
            new_board, reward, changed = game._peek(a)
            if not changed:
                continue
            empties = self._empty_cells(new_board)
            corner = self._corner_bonus(new_board)
            key = (reward, empties, corner)
            if best_tuple is None or key > best_tuple:
                best = a
                best_tuple = key
        return best if best is not None else LEFT
