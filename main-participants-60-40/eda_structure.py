"""Quick structure check: how history lines up with validation probabilities."""
from collections import defaultdict
from pathlib import Path

from slot_forecasting.io import load_inputs, read_rows, read_targets
from slot_forecasting.schema import PAYOUTS, index_history

DATA = Path("public")
queries, history = load_inputs(DATA / "validation")
indexed = index_history(queries, history)
targets = {t.query_id: t for t in read_targets(DATA / "validation/targets.csv")}
conditions = {r["query_id"]: r["condition"] for r in read_rows(DATA / "validation/conditions.csv", ("query_id", "condition"))}

by_type = defaultdict(lambda: [0.0] * 5)
by_type_n = defaultdict(int)
by_sig = defaultdict(lambda: [0.0] * 5)
by_last = defaultdict(lambda: [0.0] * 5)
by_last_n = defaultdict(int)
lens = []
gaps = []

# correlation of recent zero rate with p_0
pairs = []

for q in queries:
    t = targets[q.query_id]
    by_type[q.machine_type] = [a + b for a, b in zip(by_type[q.machine_type], t.probabilities)]
    by_type_n[q.machine_type] += 1
    by_sig[(q.machine_type, q.signal)] = [a + b for a, b in zip(by_sig[(q.machine_type, q.signal)], t.probabilities)]
    rows = indexed[(q.episode_id, q.machine_id)]
    lens.append(len(rows))
    ticks = [r.tick for r in rows]
    if len(ticks) > 1:
        gaps.append(sum(1 for a, b in zip(ticks, ticks[1:]) if b != a + 1))
    last = rows[-1].payout if rows else None
    if last is not None:
        by_last[last] = [a + b for a, b in zip(by_last[last], t.probabilities)]
        by_last_n[last] += 1
    recent = [r.payout for r in rows if r.tick >= q.cutoff - 16]
    if recent:
        z = sum(p == 0 for p in recent) / len(recent)
        pairs.append((z, t.probabilities[0], conditions[q.query_id], q.machine_type))

print("n", len(queries), "hist len min/max", min(lens), max(lens), "mean", sum(lens) / len(lens))
print("machines with any gap", sum(g > 0 for g in gaps), "of", len(gaps))

def show(title, store, counts=None):
    print("\n" + title)
    for k, v in sorted(store.items(), key=lambda kv: str(kv[0])):
        n = counts[k] if counts else by_type_n.get(k, None)
        # recompute n if not given
        print(k, [round(x, 4) for x in v])

# fix averages
print("\nBY TYPE")
for k, v in sorted(by_type.items()):
    n = by_type_n[k]
    print(k, n, [round(x / n, 4) for x in v])

print("\nBY TYPE|SIGNAL")
sig_n = defaultdict(int)
# recount
sig_sum = defaultdict(lambda: [0.0] * 5)
for q in queries:
    t = targets[q.query_id]
    key = (q.machine_type, q.signal)
    sig_n[key] += 1
    sig_sum[key] = [a + b for a, b in zip(sig_sum[key], t.probabilities)]
for k in sorted(sig_sum):
    n = sig_n[k]
    print(k, n, [round(x / n, 4) for x in sig_sum[k]])

print("\nBY LAST PAYOUT")
for k in PAYOUTS:
    if by_last_n[k]:
        print(k, by_last_n[k], [round(x / by_last_n[k], 4) for x in by_last[k]])

# binned recent zero rate vs p0
print("\nRECENT16 ZERO RATE vs mean p0")
bins = defaultdict(list)
for z, p0, cond, typ in pairs:
    bins[round(z * 5) / 5].append(p0)
for k in sorted(bins):
    xs = bins[k]
    print(f"  z~{k:.2f} n={len(xs)} mean_p0={sum(xs)/len(xs):.3f}")

# lag agreement: P(same category) 
print("\nLAG match rate (outcome category vs lag payout category) using labels only roughly via true mode")

# Within episode: variance of p0 across machines vs across episodes
from statistics import mean, pvariance
ep = defaultdict(list)
for q in queries:
    ep[q.episode_id].append(targets[q.query_id].probabilities[0])
within = mean(pvariance(v) for v in ep.values())
allp = [targets[q.query_id].probabilities[0] for q in queries]
print("p0 overall mean", mean(allp), "var", pvariance(allp), "within-ep var", within)

# condition-wise p0 variance and recent alignment
print("\nBY CONDITION recent-zero correlation")
for cond in ("stable", "short_history", "drift", "shared_change"):
    xs = [(z, p0) for z, p0, c, typ in pairs if c == cond]
    n = len(xs)
    mz = sum(z for z, _ in xs) / n
    mp = sum(p for _, p in xs) / n
    cov = sum((z - mz) * (p - mp) for z, p in xs) / n
    vz = sum((z - mz) ** 2 for z, _ in xs) / n
    vp = sum((p - mp) ** 2 for _, p in xs) / n
    corr = cov / ((vz * vp) ** 0.5) if vz and vp else 0
    print(cond, "n", n, "corr(z16,p0)", round(corr, 3), "mean p0", round(mp, 3))
