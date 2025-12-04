import pandas as pd

def csv_to_latex(path, caption):
    df = pd.read_csv(path)
    tex = df.to_latex(index=False, float_format="%.2f")
    with open(path.replace(".csv", ".tex"), "w") as f:
        f.write(f"\\begin{{table}}[h]\n\\centering\n{tex}\n\\caption{{{caption}}}\n\\end{{table}}\n")
    print("Wrote:", path.replace(".csv", ".tex"))

if __name__ == "__main__":
    csv_to_latex("scores_random.csv", "Random Agent Performance")
    csv_to_latex("scores_greedy.csv", "Greedy Agent Performance")
    csv_to_latex("scores_expectimax_d3.csv", "Expectimax D3 Performance")
