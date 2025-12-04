import numpy as np
from game_engine import Game2048
from expectimax_agent import ExpectimaxAgent
from expectimax_tc_agent import ExpectimaxTimeControlled
from heuristics import heuristic_score
from evaluate import play_one_game

import time
import numpy as np
from game_engine import Game2048

def eval_agent(agent, n=50, seed=0):
    rng = np.random.default_rng(seed)
    scores, tiles = [], []

    for idx in range(n):
        t0 = time.perf_counter()

        s = int(rng.integers(0, 1e9))
        g = Game2048(seed=s)
        while not g.is_game_over():
            a = agent.select_action(g)
            g.step(a)

        t1 = time.perf_counter()
        elapsed = t1 - t0

        scores.append(g.score)
        tiles.append(g.max_tile())

        # Print Progress
        print(
            f"[{agent.__class__.__name__}] game {idx+1}/{n} "
            f"score={g.score} max_tile={g.max_tile()} "
            f"time={elapsed:.3f}s"
        )

    return (
        np.mean(scores),
        np.median(scores),
        max(scores),
        np.mean(np.array(tiles) >= 2048)
    )


# ---- Depth Ablation ---------------------------------------------------------

def depth_ablation(depths=(1,2,3,4)):
    print("\n=== DEPTH ABLATION ===")
    for d in depths:
        agent = ExpectimaxAgent(depth=d)
        avg, med, mx, wr = eval_agent(agent, n=1, seed=10)
        print(f"Depth {d}: avg={avg:.1f}, median={med:.1f}, best={mx}, win2048={wr:.2f}")

# ---- Heuristic Ablation -----------------------------------------------------

def heuristic_ablation():
    print("\n=== HEURISTIC ABLATION ===")
    from heuristics import count_empty, smoothness, monotonicity, corner_max, positional_score

    heuristics = {
        "empty_only": lambda b: 250*count_empty(b),
        "mono_only": lambda b: 2*monotonicity(b),
        "smooth_only": lambda b: 0.1*smoothness(b),
        "corner_only": lambda b: 50*corner_max(b),
        "pos_only": lambda b: 0.5*positional_score(b),
    }

    for name, hfun in heuristics.items():
        class CustomHeuristicAgent(ExpectimaxAgent):
            def leaf_eval(self, board): return hfun(board)

        agent = CustomHeuristicAgent(depth=3)
        avg, med, mx, wr = eval_agent(agent, n=1, seed=5)
        print(f"{name}: avg={avg:.1f}, median={med:.1f}, best={mx}, win2048={wr:.2f}")

if __name__ == "__main__":
    depth_ablation()
    heuristic_ablation()
