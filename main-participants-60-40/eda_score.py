"""Score simple history estimators on validation contest points."""
from collections import defaultdict
from pathlib import Path

from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets
from slot_forecasting.rounds import round_metadata
from slot_forecasting.schema import PAYOUTS, index_history

DATA = Path("public")
meta = round_metadata(DATA)
queries, history = load_inputs(DATA / "validation")
indexed = index_history(queries, history)
# also need episode-level index
by_ep = defaultdict(list)
for q in queries:
    by_ep[q.episode_id].append(q)
targets = read_targets(DATA / "validation/targets.csv")
groups = {r["query_id"]: r["condition"] for r in read_rows(DATA / "validation/conditions.csv", ("query_id", "condition"))}
reference = read_submission(DATA / "validation/reference.csv")
idx = {p: i for i, p in enumerate(PAYOUTS)}


def norm(counts):
    s = sum(counts)
    return tuple(c / s for c in counts)


def score(name, preds):
    result = contest_score(preds, targets, groups, reference, max_points=meta["max_points"])
    bits = " ".join(f"{c[:5]}={result['by_condition'][c]['loss_ratio']:.3f}" for c in result["by_condition"])
    print(f"{name:28} pts={result['technical_points']:.3f} R={result['mean_loss_ratio']:.3f} {bits}")


def window_model(window, strength, prior):
    preds = {}
    for q in queries:
        rows = indexed[(q.episode_id, q.machine_id)]
        use = rows if window is None else [r for r in rows if r.tick >= q.cutoff - window]
        counts = [strength * p for p in prior]
        for r in use:
            counts[idx[r.payout]] += 1
        preds[q.query_id] = norm(counts)
    return preds


def decay_model(half, strength, prior):
    preds = {}
    for q in queries:
        rows = indexed[(q.episode_id, q.machine_id)]
        counts = [strength * p for p in prior]
        for r in rows:
            age = q.cutoff - 1 - r.tick
            counts[idx[r.payout]] += 2 ** (-age / half)
        preds[q.query_id] = norm(counts)
    return preds


# global prior from validation true means is cheating if used as prior fitted on val.
# Use uniform and also a rough type prior estimated from the predictions' own history (train-free empirical).
uniform = (0.2, 0.2, 0.2, 0.2, 0.2)

print("windows")
for w in (8, 12, 16, 24, 32, 48, 64, 96, None):
    score(f"win{w}_s4", window_model(w, 4, uniform))

print("decay")
for h in (4, 8, 12, 16, 24, 32, 48, 64, 96):
    score(f"half{h}_s8", decay_model(h, 8, uniform))


def markov(half, strength):
    """P(next | last payout) from the same machine, decayed, backoff to marginal."""
    preds = {}
    for q in queries:
        rows = indexed[(q.episode_id, q.machine_id)]
        seq = {r.tick: r.payout for r in rows}
        # marginal
        marg = [strength * 0.2 for _ in PAYOUTS]
        trans = {p: [strength * 0.2 for _ in PAYOUTS] for p in PAYOUTS}
        for r in rows:
            prev = seq.get(r.tick - 1)
            age = q.cutoff - 1 - r.tick
            w = 2 ** (-age / half)
            marg[idx[r.payout]] += w
            if prev is not None:
                trans[prev][idx[r.payout]] += w
        last = seq.get(q.cutoff - 1)
        counts = trans[last] if last is not None else marg
        preds[q.query_id] = norm(counts)
    return preds


print("markov lag1")
for h in (8, 16, 32, 64, 128):
    score(f"markov1_h{h}", markov(h, 2.0))


def blend_neighbors(half, self_s, neigh_s):
    preds = {}
    # precompute decayed counts per machine
    own = {}
    for q in queries:
        rows = indexed[(q.episode_id, q.machine_id)]
        counts = [0.0] * 5
        for r in rows:
            age = q.cutoff - 1 - r.tick
            counts[idx[r.payout]] += 2 ** (-age / half)
        own[q.query_id] = counts
    for q in queries:
        counts = [self_s * 0.2]
        counts = [c + self_s * 0.2 for c in own[q.query_id]]
        others = [qq for qq in by_ep[q.episode_id] if qq.machine_id != q.machine_id and qq.machine_type == q.machine_type]
        if not others:
            others = [qq for qq in by_ep[q.episode_id] if qq.machine_id != q.machine_id]
        if others:
            add = [0.0] * 5
            for qq in others:
                for i in range(5):
                    add[i] += own[qq.query_id][i]
            scale = neigh_s / len(others)
            for i in range(5):
                counts[i] += scale * add[i]
        preds[q.query_id] = norm(counts)
    return preds


print("neighbor blend")
for h, ns in ((16, 0.15), (24, 0.25), (32, 0.3), (16, 0.0)):
    score(f"neigh_h{h}_n{ns}", blend_neighbors(h, 4, ns))
