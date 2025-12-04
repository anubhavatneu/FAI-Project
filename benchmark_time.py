from game_engine import Game2048, ACTION_NAMES
from expectimax_tc_agent import ExpectimaxTimeControlled

def run(seed=0, budget_ms=50):
    g = Game2048(seed=seed)
    a = ExpectimaxTimeControlled(timer_budget_sec=budget_ms/1000.0, empty_cell_cap=8)
    moves = 0
    while not g.is_game_over():
        act = a.select_action(g)
        _, r, done, _ = g.step(act)
        moves += 1
        if done: break
    print({
        "seed": seed,
        "budget_ms": budget_ms,
        "score": g.score,
        "max_tile": g.max_tile(),
        "moves": moves
    })

if __name__ == "__main__":
    for ms in (20, 50, 100, 1000, 2000, 5000):
        run(seed=42, budget_ms=ms)
