import csv
from agents import RandomAgent, GreedyImmediateAgent
from evaluate import evaluate_agent
from expectimax_agent import ExpectimaxAgent   
import numpy as np

def print_summary(title, res):
    print(f"\n=== {title} ===")
    print(f"Games:          {res['games']}")
    print(f"Avg score:      {res['avg_score']:.1f}")
    print(f"Median score:   {res['median_score']:.1f}")
    print(f"Best score:     {res['best_score']}")
    print(f"Win rate 2048:  {res['win_rate_2048']*100:.2f}%")
    print(f"Best tile hist: {res['best_tile_hist']}")

def save_csv(path, rows):
    if not rows: return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved CSV -> {path}")

def collect_scores(agent, n_games=200, seed=123):
    rng = np.random.default_rng(seed)
    from evaluate import play_one_game
    per_game = []
    for idx in range(n_games):
        s = int(rng.integers(0, 1_000_000_000))
        out = play_one_game(agent, seed=s, verbose=False)
        per_game.append({"score": out["score"], "max_tile": out["max_tile"]})
        print(f"[{agent.__class__.__name__}] finished game {idx+1}/{n_games}")
    return per_game



if __name__ == "__main__":
    N = 30

    res_random = evaluate_agent(RandomAgent(), n_games=N, seed=7)
    res_greedy = evaluate_agent(GreedyImmediateAgent(), n_games=N, seed=7)

    ex = ExpectimaxAgent(depth=2, empty_cell_cap=6)
    res_ex = evaluate_agent(ex, n_games=N, seed=7)

    print_summary("RandomAgent", res_random)
    print_summary("GreedyImmediateAgent", res_greedy)
    print_summary("Expectimax(depth=3)", res_ex)

    rows_r = collect_scores(RandomAgent(), n_games=N, seed=99);  [r.update({"agent":"random"}) for r in rows_r]
    rows_g = collect_scores(GreedyImmediateAgent(), n_games=N, seed=99);  [r.update({"agent":"greedy"}) for r in rows_g]
    rows_e = collect_scores(ExpectimaxAgent(depth=3, empty_cell_cap=8), n_games=N, seed=99);  [r.update({"agent":"expectimax_d3"}) for r in rows_e]

    save_csv("scores_random.csv", rows_r)
    save_csv("scores_greedy.csv", rows_g)
    save_csv("scores_expectimax_d3.csv", rows_e)
