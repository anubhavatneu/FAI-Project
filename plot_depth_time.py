import matplotlib.pyplot as plt
from expectimax_tc_agent import ExpectimaxTimeControlled
from game_engine import Game2048

def profile(budget_ms=50, seed=42):
    g = Game2048(seed=seed)
    a = ExpectimaxTimeControlled(timer_budget_sec=budget_ms/1000, empty_cell_cap=8)
    act = a.select_action(g)
    return a.stats.max_depth_reached, a.stats.nodes

def main():
    budgets = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]

    depths, nodes = [], []

    for b in budgets:
        d, n = profile(b)
        depths.append(d)
        nodes.append(n)

    plt.figure()
    plt.plot(budgets, depths, marker='o')
    plt.xlabel("Time per Move (ms)")
    plt.ylabel("Max Depth Reached")
    plt.title("Depth vs Time Budget")
    plt.savefig("depth_vs_time.png")

    plt.figure()
    plt.plot(budgets, nodes, marker='s')
    plt.xlabel("Time per Move (ms)")
    plt.ylabel("Nodes Expanded")
    plt.title("Nodes vs Time Budget")
    plt.savefig("nodes_vs_time.png")

    print("Saved: depth_vs_time.png, nodes_vs_time.png")

if __name__ == "__main__":
    main()
