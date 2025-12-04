import numpy as np
from typing import Dict, Any
from game_engine import Game2048
from agents import RandomAgent, GreedyImmediateAgent

def play_one_game(agent, seed=None, verbose=False) -> Dict[str, Any]:
    game = Game2048(seed=seed)
    if verbose:
        game.render()
    while not game.is_game_over():
        action = agent.select_action(game)
        board, reward, done, info = game.step(action)
        if verbose:
            print(f"Move reward: {reward}")
            game.render()
        if done:
            break
    return {"score": game.score, "max_tile": game.max_tile()}

def evaluate_agent(agent, n_games=100, seed=123) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    scores, max_tiles = [], []
    for _ in range(n_games):
        s = int(rng.integers(0, 1_000_000_000))
        result = play_one_game(agent, seed=s)
        scores.append(result["score"])
        max_tiles.append(result["max_tile"])
    scores, max_tiles = np.array(scores), np.array(max_tiles)
    return {
        "games": n_games,
        "avg_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "best_score": int(np.max(scores)),
        "win_rate_2048": float(np.sum(max_tiles >= 2048)) / n_games,
        "best_tile_hist": {int(v): int(np.sum(max_tiles == v)) for v in np.unique(max_tiles)},
    }

if __name__ == "__main__":
    print("RandomAgent results:")
    ra = RandomAgent()
    print(evaluate_agent(ra, n_games=20, seed=42))

    print("\nGreedyImmediateAgent results:")
    ga = GreedyImmediateAgent()
    print(evaluate_agent(ga, n_games=20, seed=42))
