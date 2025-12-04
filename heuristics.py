import numpy as np

POSITION_MASK = np.array([
    [16, 15, 14, 13],
    [ 9, 10, 11, 12],
    [ 8,  7,  6,  5],
    [ 1,  2,  3,  4],
], dtype=np.float64)

def count_empty(board: np.ndarray) -> int:
    return int(np.sum(board == 0))

def corner_max(board: np.ndarray) -> int:
    """1 if the maximum tile is in a corner; else 0."""
    m = board.max()
    corners = (board[0,0], board[0,3], board[3,0], board[3,3])
    return 1 if m in corners else 0

def smoothness(board: np.ndarray) -> float:
    """
    Negative sum of absolute diffs between neighbors (4-neighborhood).
    Lower diffs => smoother => higher (less negative) score.
    """
    b = board.astype(np.float64)
    s = 0.0
    # horizontal diffs
    s -= np.sum(np.abs(b[:, 1:] - b[:, :-1]))
    # vertical diffs
    s -= np.sum(np.abs(b[1:, :] - b[:-1, :]))
    return float(s)

def monotonicity(board: np.ndarray) -> float:
    """
    Measures how consistently values increase/decrease across rows/cols.
    We reward monotone sequences (either direction).
    """
    b = board.astype(np.float64)
    score = 0.0

    score += np.sum(b[:, :-1] <= b[:, 1:])  
    score += np.sum(b[:, :-1] >= b[:, 1:])  

    score += np.sum(b[:-1, :] <= b[1:, :])
    score += np.sum(b[:-1, :] >= b[1:, :])
    return float(score)

def positional_score(board: np.ndarray) -> float:
    return float(np.sum(POSITION_MASK * board.astype(np.float64)))

def heuristic_score(board: np.ndarray, weights=None) -> float:
    """
    Composite evaluator for leaf nodes.
    Default weights are reasonable starting points; we'll tune later.
    """
    if weights is None:
        weights = {
            "empty":  250.0,     
            "mono":    2.0,
            "smooth":  0.1,      
            "corner": 50.0,
            "pos":     0.5,
        }
    e  = count_empty(board)
    m  = monotonicity(board)
    sm = smoothness(board)
    c  = corner_max(board)
    p  = positional_score(board)
    return (weights["empty"] * e
          + weights["mono"]  * m
          + weights["smooth"]* sm
          + weights["corner"]* c
          + weights["pos"]   * p)
