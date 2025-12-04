import time
from dataclasses import dataclass, field

class Timer:
    def __init__(self, budget_sec: float):
        self.budget = float(budget_sec)
        self.start = None
    def start_now(self):
        self.start = time.perf_counter()
    def elapsed(self) -> float:
        if self.start is None: return 0.0
        return time.perf_counter() - self.start
    def time_left(self) -> float:
        return max(0.0, self.budget - self.elapsed())
    def expired(self) -> bool:
        return self.elapsed() >= self.budget

@dataclass
class SearchStats:
    nodes: int = 0
    max_depth_reached: int = 0
    tt_hits: int = 0
    tt_puts: int = 0
    def bump(self, depth: int):
        self.nodes += 1
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth
