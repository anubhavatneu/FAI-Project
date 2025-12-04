import csv
import matplotlib.pyplot as plt

def read_scores(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"score": int(r["score"]), "max_tile": int(r["max_tile"]), "agent": r["agent"]})
    return rows

def plot_histogram(scores, title, out_path):
    plt.figure()
    plt.hist(scores, bins=30)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot -> {out_path}")

def plot_best_tile_hist(max_tiles, title, out_path):
    # frequency by tile
    freq = {}
    for t in max_tiles:
        freq[t] = freq.get(t, 0) + 1
    xs = sorted(freq.keys())
    ys = [freq[x] for x in xs]
    plt.figure()
    plt.bar([str(x) for x in xs], ys)
    plt.title(title)
    plt.xlabel("Best Tile")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot -> {out_path}")

if __name__ == "__main__":
    r = read_scores("scores_random.csv")
    g = read_scores("scores_greedy.csv")
    v = read_scores("scores_expectimax_d3.csv")

    plot_histogram([x["score"] for x in r], "RandomAgent Score Distribution", "hist_random_scores.png")
    plot_histogram([x["score"] for x in g], "GreedyImmediateAgent Score Distribution", "hist_greedy_scores.png")
    plot_histogram([x["score"] for x in v], "ExpectimaxAgent Score Distribution", "hist_expectimax_scores.png")


    plot_best_tile_hist([x["max_tile"] for x in r], "RandomAgent Best Tiles", "bar_random_best_tile.png")
    plot_best_tile_hist([x["max_tile"] for x in g], "GreedyImmediateAgent Best Tiles", "bar_greedy_best_tile.png")
    plot_best_tile_hist([x["max_tile"] for x in v], "ExpectimaxAgent Best Tiles", "bar_expectimax_best_tile.png")

