"""Are lag effects shared, or do they flip by machine?"""
from collections import defaultdict
from pathlib import Path

from slot_forecasting.io import load_inputs, read_labels
from slot_forecasting.schema import PAYOUTS, index_history

queries, history = load_inputs(Path("public/train"))
labels = read_labels(Path("public/train/labels.csv"), queries)
indexed = index_history(queries, history)
idx = {p: i for i, p in enumerate(PAYOUTS)}

# global counts of (lag, prev, next)
# use all consecutive observations plus label at cutoff
lag_counts = defaultdict(lambda: [[0, 0] for _ in range(5)])  # [n, n_zero_next] wait full 5
# store [5] counts
lag_counts = defaultdict(lambda: [0] * 5)
base = [0] * 5
per_machine_sign = defaultdict(list)  # lag -> list of machine correlations

for q in queries:
    rows = list(indexed[(q.episode_id, q.machine_id)])
    seq = {r.tick: r.payout for r in rows}
    events = [(r.tick, r.payout) for r in rows]
    events.append((q.cutoff, labels[q.query_id]))
    # machine-level: for lag 1, count next zero given prev zero vs not, using events except we need prev from seq only before
    z_given_z = [0, 0]
    z_given_nz = [0, 0]
    for tick, pay in events:
        base[idx[pay]] += 1
        for lag in range(1, 9):
            prev = seq.get(tick - lag)
            if prev is None:
                continue
            lag_counts[(lag, prev)][idx[pay]] += 1
        prev1 = seq.get(tick - 1)
        if prev1 is None:
            continue
        if prev1 == 0:
            z_given_z[0] += 1
            z_given_z[1] += int(pay == 0)
        else:
            z_given_nz[0] += 1
            z_given_nz[1] += int(pay == 0)
    if z_given_z[0] >= 20 and z_given_nz[0] >= 20:
        a = z_given_z[1] / z_given_z[0]
        b = z_given_nz[1] / z_given_nz[0]
        per_machine_sign[q.machine_type].append(a - b)

print("base", [round(c / sum(base), 3) for c in base], "n", sum(base))
print("\nP(next=0 | lag payout) / base0")
b0 = base[0] / sum(base)
for lag in range(1, 9):
    bits = []
    for prev in PAYOUTS:
        c = lag_counts[(lag, prev)]
        n = sum(c)
        if n:
            bits.append(f"{prev}:{c[0]/n:.3f}(n={n})")
    print(f"lag {lag}", " ".join(bits), "base", round(b0, 3))

print("\nPer-machine lift P(0|prev0)-P(0|prev!=0), lag1, n machines with enough data")
for typ, lifts in per_machine_sign.items():
    pos = sum(x > 0.02 for x in lifts)
    neg = sum(x < -0.02 for x in lifts)
    mid = len(lifts) - pos - neg
    avg = sum(lifts) / len(lifts)
    print(typ, "n", len(lifts), "avg", round(avg, 3), "pos", pos, "neg", neg, "flat", mid)
