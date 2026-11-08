"""Isolate the lag-7 effect from a sticky base rate."""
from collections import defaultdict
from pathlib import Path

from slot_forecasting.io import load_inputs, read_labels
from slot_forecasting.schema import PAYOUTS, index_history

queries, history = load_inputs(Path("public/train"))
labels = read_labels(Path("public/train/labels.csv"), queries)
indexed = index_history(queries, history)

# For each event, recent zero rate in previous 16 ticks (observed only), and lag7 payout.
cells = defaultdict(lambda: [0] * 5)
idx = {p: i for i, p in enumerate(PAYOUTS)}

# also full transition lag7 by type
by_type = defaultdict(lambda: [0] * 5)

for q in queries:
    rows = indexed[(q.episode_id, q.machine_id)]
    seq = {r.tick: r.payout for r in rows}
    events = [(r.tick, r.payout) for r in rows]
    events.append((q.cutoff, labels[q.query_id]))
    ticks_sorted = [r.tick for r in rows]
    for tick, pay in events:
        prevs = [seq[t] for t in range(tick - 16, tick) if t in seq]
        if len(prevs) < 8:
            continue
        z = sum(p == 0 for p in prevs) / len(prevs)
        bucket = min(4, int(z * 5))
        lag7 = seq.get(tick - 7)
        key_lag = "miss" if lag7 is None else str(lag7)
        cells[(bucket, key_lag)][idx[pay]] += 1
        if lag7 is not None:
            by_type[(q.machine_type, lag7)][idx[pay]] += 1

print("P(next) by recent-zero bucket x lag7")
for bucket in range(5):
    print(f"\nzero-rate bucket {bucket}")
    for lag in ("0", "1", "2", "5", "20", "miss"):
        c = cells[(bucket, lag)]
        n = sum(c)
        if not n:
            continue
        print(f"  lag7={lag:4} n={n:6} " + " ".join(f"{p}:{c[i]/n:.3f}" for i, p in enumerate(PAYOUTS)))

print("\nLag7 by machine type, P(next)")
for typ in ("steady", "balanced", "jackpot"):
    print("\n", typ)
    for lag in PAYOUTS:
        c = by_type[(typ, lag)]
        n = sum(c)
        print(f"  {lag:2} n={n:6} " + " ".join(f"{p}:{c[i]/n:.3f}" for i, p in enumerate(PAYOUTS)))
