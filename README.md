
---

# 📘 **2048 AI Agent — FAI Project**

### *Expectimax Search, Heuristics, Time-Controlled Planning & Ablations*

Author: **Anubhav Tiwari**  
Course: *Foundations of Artificial Intelligence*  
Semester: *Fall 2025*

---

 ## **1. Overview**

This project develops a complete **AI system for playing the 2048 puzzle game**, implementing:

* A fully functional **2048 Game Engine**
* Baseline Agents (Random, Greedy)
* A **Depth-limited Expectimax AI** with Heuristics
* A **Time-controlled Expectimax AI** using Iterative Deepening
* **Ablation Studies** (Heuristic & Depth)
* **Benchmarking** (Score, Max-Tile Distribution, Win-Rate)
* **Plotting** Depth vs Time, Nodes vs Time
* Modular, Extensible Architecture

The AI consistently outperforms baseline agents and demonstrates the role of search depth, stochastic branching, heuristics, and computation budget.

---

 ## **2. Project Structure**

```
.
│── game_engine.py
│── agents.py
│── expectimax_agent.py
│── expectimax_tc_agent.py
│── heuristics.py
│── benchmark.py
│── benchmark_time.py
│── ablations.py
│── plot_results.py
│── plot_depth_time.py
│── scores_random.csv
│── scores_greedy.csv
│── scores_expectimax_d3.csv
│
├── Utility / Optional Modules
│   ├── search_utils.py
│   ├── orderings.py
│   ├── value_estimator.py
│   ├── profile_search.py
│   ├── test_moves.py
│   ├── cli.py
│   └── evaluate.py
│
└── README.md   
```

---

 ## **3. Game Engine**

### **`game_engine.py`**

Implements a deterministic and correct 2048 simulator:

* `_slide_and_merge` — Merges tiles according to game rules.
* `_apply_move` — Applies directional moves.
* `_spawn_tile` — Spawns 2-tile (90%) or 4-tile (10%).
* `legal_moves` — Checks available moves.
* `step` — Executes a full decision step.
* `is_game_over`, `max_tile`
* `render` — Pretty-Prints the Board.

This engine is the foundation for *all* AI experimentation in this project.

---

 ## **4. Baseline Agents**

### **`agents.py`**

| Agent                    | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| **RandomAgent**          | Makes Uniform Random Valid Move                            |
| **GreedyImmediateAgent** | Chooses Action which gives Maximum Immediate Merge Reward |

These provide reference performance levels for comparison.

---

 ## **5. Expectimax AI (Depth-Limited)**

### **`expectimax_agent.py`**

A classical **Expectimax** search agent:

* Max-nodes → Chooses Best Move.
* Chance-nodes → Spawns 2 or 4 with Probabilities (0.9, 0.1).
* Depth-Limited Recursion.
* Transposition Table Caching.
* `empty_cell_cap` → Limits Branching Explosion.

### Heuristic Evaluation

Defined in **`heuristics.py`**, a composite function:

* Empty Tiles
* Smoothness
* Monotonicity
* Corner Maximization
* Positional Weights

This evaluation guides Expectimax at leaf nodes.

---

 ## **6. Time-Controlled Expectimax (Real-Time)**

### **`expectimax_tc_agent.py`**

Implements:

* **Iterative Deepening** (depth = 1 → 2 → 3...)
* **Move Ordering** (Tries Promising Moves First)
* **Time Budget** (20ms, 50ms, 100ms...)
* **Pruned Chance Nodes**
* **Search Profiling**

This allows competitive performance even under strict deadlines.

---

 ## **7. Benchmarking Framework**



### **`benchmark.py`** — *Fixed Depth Benchmarking*

Produces:

* Average Score
* Median Score
* Max Tile Distribution
* Win Rate
* **CSV Outputs for Plotting**

---

### **`benchmark_time.py`** — *Time-based Benchmarking*

For testing seeds under different time budgets.

---
### **`evaluate.py`**

Central evaluation functions: 

* `play_one_game(agent)`
* `evaluate_agent(agent, n_games)`

---


 ## **8. Ablation Studies**

### **`ablations.py`**

Two types of Ablations -


### **A. Depth Ablation**

Tests difficulty vs depth.

Has Clear **exponential** cost.

---

### **B. Heuristic Ablation**

Tests removing one feature at a time:

* Removing *Empty Tiles* → High Congestion
* Removing *Monotonicity* → Chaotic Tile Placement
* Removing *Smoothness* → Unstable Boards
* Removing *Corner Weighting* → Frequent Collapses
* Removing *Positional Score* → Weaker Long-Term Planning

---

 ## **9. Plotting Depth vs Time & Nodes vs Time**

### **`plot_depth_time.py`** generated:

#### **Depth vs Time Curve**

Shows how computation time grows exponentially per depth.

#### **Nodes Expanded vs Time Curve**

Shows how node count also grows Exponentially → Branching factor ≈ 4 × (#empties × 2 tiles).

This empirically validates Expectimax complexity.

---

 ## **10. Plotting Score and Tile Distributions**

### **`plot_results.py`** generated two sets of visualizations summarizing the empirical performance of the three agents.

---

 ### **A. Score Histograms**

For each agent, the script plots a histogram of final scores over 30 games.

* **RandomAgent**: Scores cluster tightly between 500–1500, reflecting consistently weak performance.
* **GreedyImmediateAgent**: Shows a wider spread (≈ 1500–8000); strong early merges but unstable late-game behavior.
* **Expectimax (Depth 3)**: Displays a broad, right-skewed distribution (≈ 4000–28000), indicating far stronger and more varied gameplay.

These histograms illustrate how planning depth and heuristic evaluation dramatically increase both average performance and upper-end potential.

---

 ### **B. Best Tile Distributions**

Bar charts show the frequency of the highest tile achieved in each game.

* **RandomAgent**: Mostly ends at 64 or 128.
* **GreedyImmediateAgent**: Commonly reaches 256 or 512.
* **Expectimax (Depth 3)**: Frequently reaches 512 or 1024, with occasional 2048 runs.

These plots visually confirm that Expectimax consistently progresses deeper into the game and produces much higher-value outcomes.

---

 ### **Summary**

Together, the score histograms and tile-frequency charts provide a clear performance comparison:
**Expectimax strongly outperforms both baselines in consistency, depth of play, and maximum achievable tiles.**

---


 ## **11. Full Module Index**

### **Core Modules**

* `game_engine.py` — Full 2048 Game Implementation
* `agents.py` — Random & Greedy
* `expectimax_agent.py` — Depth-Limited Expectimax
* `expectimax_tc_agent.py` — Time-controlled Expectimax
* `heuristics.py` — Heuristic Functions

### **Evaluation**

* `benchmark.py`
* `benchmark_time.py`
* `ablations.py`

### **Plotting**

* `plot_results.py`
* `plot_depth_time.py`

### **Utilities**

* `search_utils.py`
* `orderings.py`
* `profile_search.py`
* `test_moves.py`
* `cli.py`
* `value_estimator.py`
* `evaluate.py`
---

 ## **12. Future Work Prospects**

* Train a supervised **Neural Value Estimator**
* Hybrid **Expectimax + Monte Carlo Tree Search**
* Real-time Graphical **GUI**
* Deep RL: DQN / TD-learning version of 2048

---

 ## **13. How to Run**

```bash
pip install -r requirements.txt
python benchmark.py
python plot_results.py
python benchmark_time.py
python ablations.py
python plot_depth_time.py
```

---
