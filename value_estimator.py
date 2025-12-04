import numpy as np
import joblib as _joblib  
from typing import List, Tuple
from game_engine import Game2048
from expectimax_tc_agent import ExpectimaxTimeControlled
from heuristics import heuristic_score

def board_features(b: np.ndarray) -> np.ndarray:
    x = b.flatten().astype(np.float64)
    x_log = np.where(x > 0, np.log2(x), 0.0)
    empty = np.array([np.sum(b == 0)], dtype=np.float64)
    return np.concatenate([x, x_log, empty], axis=0)

def generate_dataset(n_states=2000, seed=0, teacher_budget_ms=50):
    rng = np.random.default_rng(seed)
    X, y = [], []
    teacher = ExpectimaxTimeControlled(timer_budget_sec=teacher_budget_ms/1000.0, empty_cell_cap=8)
    for k in range(n_states):
        s = int(rng.integers(0, 1_000_000_000))
        g = Game2048(seed=s)
        steps = int(rng.integers(2, 12))
        for _ in range(steps):
            if g.is_game_over(): break
            from agents import RandomAgent
            a = RandomAgent(rng=rng).select_action(g)
            g.step(a)

        board = g.board.copy()
        _ = teacher.select_action(g)  
        val = heuristic_score(board)
        X.append(board_features(board)); y.append(val)
    X = np.stack(X, axis=0); y = np.array(y, dtype=np.float64)
    np.savez("value_data.npz", X=X, y=y)
    print("Saved value_data.npz with", X.shape, "features")
    return X, y

def train_ridge(X, y, alpha=1.0):
    try:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha).fit(X, y)
        _joblib.dump(model, "value_model.joblib")
        print("Saved value_model.joblib")
        return model
    except Exception as e:
        print("sklearn not available or failed:", e)
        print("Saving raw data only; you can train later.")
        return None

if __name__ == "__main__":
    X, y = generate_dataset(n_states=1000, seed=0, teacher_budget_ms=30)
    train_ridge(X, y, alpha=1.0)
