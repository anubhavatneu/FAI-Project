import argparse
from game_engine import Game2048, ACTION_NAMES, UP, DOWN, LEFT, RIGHT
from agents import RandomAgent, GreedyImmediateAgent

KEY_TO_ACTION = {"w": UP, "s": DOWN, "a": LEFT, "d": RIGHT}

def human_loop(seed=None):
    game = Game2048(seed=seed)
    print("Play with W/A/S/D. Press Ctrl+C to quit.")
    game.render()
    while not game.is_game_over():
        key = input("Move> ").strip().lower()
        if key not in KEY_TO_ACTION:
            print("Invalid key.")
            continue
        _, reward, done, _ = game.step(KEY_TO_ACTION[key])
        print(f"Reward: {reward}")
        game.render()
        if done:
            print("Game over!")
            break

def agent_loop(agent_name="greedy", seed=None, verbose=False):
    agent = RandomAgent() if agent_name == "random" else GreedyImmediateAgent()
    game = Game2048(seed=seed)
    if verbose:
        game.render()
    while not game.is_game_over():
        action = agent.select_action(game)
        _, reward, done, _ = game.step(action)
        if verbose:
            print(f"{agent_name.title()} chose {ACTION_NAMES[action]} | reward {reward}")
            game.render()
        if done:
            break
    print(f"Final score: {game.score}, Max tile: {game.max_tile()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "agent"], default="agent")
    parser.add_argument("--agent", choices=["random", "greedy"], default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "human":
        human_loop(seed=args.seed)
    else:
        agent_loop(agent_name=args.agent, seed=args.seed, verbose=args.verbose)
