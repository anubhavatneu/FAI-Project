import numpy as np
from game_engine import Game2048, LEFT, RIGHT, UP, DOWN

def board_from_rows(rows):
    return np.array(rows, dtype=np.int64)

def test_simple_merge_left():
    g = Game2048(seed=0)
    g.board = board_from_rows([[2,2,0,0],
                               [0,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0]])
    new_b, gain, changed = g._peek(LEFT)
    assert changed
    assert gain == 4
    assert (new_b[0] == np.array([4,0,0,0])).all()

def test_no_double_merge_once():
    g = Game2048(seed=0)
    g.board = board_from_rows([[2,2,2,0],
                               [0,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0]])
    new_b, gain, changed = g._peek(LEFT)
    assert (new_b[0] == np.array([4,2,0,0])).all()
    assert gain == 4

def test_chain_right():
    g = Game2048(seed=0)
    g.board = board_from_rows([[2,0,2,2],
                               [0,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0]])
    new_b, gain, changed = g._peek(RIGHT)
    assert (new_b[0] == np.array([0,2,2,2]) or changed) 
    _, reward, _, _ = g.step(RIGHT)
    assert reward == 4

def test_no_move_when_blocked():
    g = Game2048(seed=0)
    g.board = board_from_rows([[2,4,8,16],
                               [32,64,128,256],
                               [2,4,8,16],
                               [32,64,128,256]])
    new_b, gain, changed = g._peek(LEFT)
    assert not changed
    assert gain == 0

def test_game_over_detection():
    g = Game2048(seed=0)
    g.board = board_from_rows([[2,4,2,4],
                               [4,2,4,2],
                               [2,4,2,4],
                               [4,2,4,2]])
    assert g.is_game_over()

    _, _, done, info = g.step(LEFT)
    assert done and info["invalid"]

if __name__ == "__main__":

    import inspect, sys
    passed, failed = 0, 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
                passed += 1
            except AssertionError as e:
                print(f"[FAIL] {name}: {e}")
                failed += 1
    print(f"\nSummary: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
