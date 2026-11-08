"""Hierarchical payout histogram with backoff. Train on past spins only."""
from collections import defaultdict
from pathlib import Path

from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_submission
from slot_forecasting.rounds import round_metadata
from slot_forecasting.schema import PAYOUTS, index_history

PAY_I = {p: i for i, p in enumerate(PAYOUTS)}


def sequences(queries, history):
    indexed = index_history(queries, history)
    return {key: {row.tick: row.payout for row in rows} for key, rows in indexed.items()}


def features(seq, tick, machine_type, signal):
    recent = [seq[t] for t in range(tick - 16, tick) if t in seq]
    n = len(recent)
    z = (sum(p == 0 for p in recent) / n) if n else 0.5
    bucket = min(4, int(z * 5))
    # coarser long window
    long = [seq[t] for t in range(tick - 64, tick) if t in seq]
    ln = len(long)
    zl = (sum(p == 0 for p in long) / ln) if ln else z
    long_bucket = min(4, int(zl * 5))
    lag1 = seq.get(tick - 1, -1)
    lag7 = seq.get(tick - 7, -1)
    lag8 = seq.get(tick - 8, -1)
    return {
        "type": machine_type,
        "signal": signal,
        "z": bucket,
        "zl": long_bucket,
        "lag1": lag1,
        "lag7": lag7,
        "lag8": lag8,
        "n": n,
        "short": int(n < 12),
    }


def add_counts(store, key, payout):
    row = store[key]
    row[PAY_I[payout]] += 1


def fit_counts(queries, history, labels):
    seqs = sequences(queries, history)
    levels = [defaultdict(lambda: [0] * 5) for _ in range(6)]
    for q in queries:
        seq = seqs[(q.episode_id, q.machine_id)]
        events = list(seq.items())
        events.append((q.cutoff, labels[q.query_id]))
        for tick, pay in events:
            if tick < 32:
                continue
            f = features(seq, tick, q.machine_type, q.signal)
            keys = (
                ("all",),
                ("t", f["type"]),
                ("tz", f["type"], f["z"]),
                ("tz7", f["type"], f["z"], f["lag7"]),
                ("tz17", f["type"], f["z"], f["lag7"], f["lag1"]),
                ("tzs7", f["type"], f["signal"], f["z"], f["lag7"]),
            )
            for level, key in zip(levels, keys):
                add_counts(level, key, pay)
    return levels


def mix(levels, f, strengths):
    keys = (
        ("all",),
        ("t", f["type"]),
        ("tz", f["type"], f["z"]),
        ("tz7", f["type"], f["z"], f["lag7"]),
        ("tz17", f["type"], f["z"], f["lag7"], f["lag1"]),
        ("tzs7", f["type"], f["signal"], f["z"], f["lag7"]),
    )
    counts = [1.0] * 5  # tiny floor
    for level, key, strength in zip(levels, keys, strengths):
        row = level.get(key)
        if not row:
            continue
        n = sum(row)
        if n <= 0:
            continue
        # add this level's counts, shrunk by leaving parent mass = strength
        for i in range(5):
            counts[i] = counts[i] * (strength / (strength + n)) * (sum(counts) and 1 or 1)
        # simpler: posterior combination
    # redo with a clean chain
    dist = [0.2] * 5
    for level, key, strength in zip(levels, keys, strengths):
        row = level.get(key)
        n = sum(row) if row else 0
        if n <= 0:
            continue
        dist = [(strength * d + c) / (strength + n) for d, c in zip(dist, row)]
    total = sum(dist)
    return tuple(v / total for v in dist)


def predict(levels, queries, history, strengths):
    seqs = sequences(queries, history)
    out = {}
    for q in queries:
        seq = seqs.get((q.episode_id, q.machine_id), {})
        f = features(seq, q.cutoff, q.machine_type, q.signal)
        out[q.query_id] = mix(levels, f, strengths)
    return out


def evaluate(preds):
    data = Path("public")
    meta = round_metadata(data)
    targets = read_targets(data / "validation/targets.csv")
    groups = {r["query_id"]: r["condition"] for r in read_rows(data / "validation/conditions.csv", ("query_id", "condition"))}
    reference = read_submission(data / "validation/reference.csv")
    result = contest_score(preds, targets, groups, reference, max_points=meta["max_points"])
    bits = " ".join(f"{c[:5]}={result['by_condition'][c]['loss_ratio']:.3f}" for c in result["by_condition"])
    print(f"pts={result['technical_points']:.3f} R={result['mean_loss_ratio']:.3f} {bits}")
    return result


if __name__ == "__main__":
    train_q, train_h = load_inputs(Path("public/train"))
    labels = read_labels(Path("public/train/labels.csv"), train_q)
    print("fitting")
    levels = fit_counts(train_q, train_h, labels)
    val_q, val_h = load_inputs(Path("public/validation"))
    grids = [
        (40, 30, 20, 12, 8, 12),
        (20, 16, 12, 8, 6, 8),
        (80, 40, 24, 16, 10, 16),
        (30, 20, 10, 6, 4, 8),
        (50, 40, 30, 20, 15, 20),
        (15, 12, 8, 5, 3, 6),
    ]
    for strengths in grids:
        print(strengths, end=" ")
        evaluate(predict(levels, val_q, val_h, strengths))
