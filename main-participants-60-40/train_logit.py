"""Multinomial logistic forecast from pre-cutoff history features."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets
from slot_forecasting.rounds import round_metadata
from slot_forecasting.schema import PAYOUTS

PAY_I = {p: i for i, p in enumerate(PAYOUTS)}
TYPES = ("steady", "balanced", "jackpot")
WINDOWS = (8, 16, 32, 64, 128)
LAGS = (1, 2, 3, 4, 5, 6, 7, 8)


def pack(queries, history):
    machines = []
    meta = {}
    for q in queries:
        key = (q.episode_id, q.machine_id)
        meta[key] = q
        machines.append(key)
    # preserve episode groups
    episodes = {}
    for key in machines:
        episodes.setdefault(key[0], []).append(key)
    max_tick = 0
    rows = []
    for play in history:
        rows.append(play)
        if play.tick > max_tick:
            max_tick = play.tick
    width = max_tick + 2
    grids = {}
    for key in machines:
        grids[key] = np.full(width, -1, dtype=np.int8)
    for play in history:
        grids[(play.episode_id, play.machine_id)][play.tick] = PAY_I[play.payout]
    return meta, episodes, grids, width


def _window_fracs(grid, tick, window):
    start = max(0, tick - window)
    sl = grid[start:tick]
    sl = sl[sl >= 0]
    n = sl.size
    fr = np.zeros(5, dtype=np.float64)
    if n:
        for i in range(5):
            fr[i] = np.sum(sl == i) / n
    return fr, n / window


def row_features(grid, tick, q, episodes, grids):
    feats = []
    for typ in TYPES:
        feats.append(1.0 if q.machine_type == typ else 0.0)
    for sig in (-1, 0, 1):
        feats.append(1.0 if q.signal == sig else 0.0)
    fracs = {}
    for window in WINDOWS:
        fr, cov = _window_fracs(grid, tick, window)
        fracs[window] = fr
        feats.extend(fr.tolist())
        feats.append(cov)
    feats.append(fracs[8][0] - fracs[64][0])
    feats.append(fracs[16][0] - fracs[128][0])
    feats.append(fracs[8][1] - fracs[64][1])
    for lag in LAGS:
        pos = tick - lag
        val = int(grid[pos]) if pos >= 0 else -1
        for i in range(5):
            feats.append(1.0 if val == i else 0.0)
        feats.append(1.0 if val < 0 else 0.0)
    # lag7 x type
    pos = tick - 7
    val = int(grid[pos]) if pos >= 0 else -1
    for typ in TYPES:
        typ_on = 1.0 if q.machine_type == typ else 0.0
        for i in range(5):
            feats.append(typ_on if val == i else 0.0)
        feats.append(typ_on if val < 0 else 0.0)
    # neighbors
    same = np.zeros(5, dtype=np.float64)
    anyf = np.zeros(5, dtype=np.float64)
    ns = na = 0
    for other in episodes[q.episode_id]:
        if other == (q.episode_id, q.machine_id):
            continue
        fr, _ = _window_fracs(grids[other], tick, 16)
        anyf += fr
        na += 1
        if meta_type(grids, other, q) if False else grids and False:
            pass
    return feats  # placeholder replaced below


def build_xy(queries, history, labels=None, min_tick=48):
    meta, episodes, grids, width = pack(queries, history)
    # type lookup
    types = {key: q.machine_type for key, q in meta.items()}
    xs = []
    ys = []
    qids = []

    def features(key, tick):
        q = meta[key]
        grid = grids[key]
        feats = []
        for typ in TYPES:
            feats.append(1.0 if q.machine_type == typ else 0.0)
        for sig in (-1, 0, 1):
            feats.append(1.0 if q.signal == sig else 0.0)
        fracs = {}
        for window in WINDOWS:
            fr, cov = _window_fracs(grid, tick, window)
            fracs[window] = fr
            feats.extend(fr.tolist())
            feats.append(cov)
        feats.append(float(fracs[8][0] - fracs[64][0]))
        feats.append(float(fracs[16][0] - fracs[128][0]))
        observed = grid[:tick]
        observed = observed[observed >= 0]
        feats.append(np.log1p(observed.size))
        # missingness in last 40
        start = max(0, tick - 40)
        span = grid[start:tick]
        feats.append(float(np.mean(span < 0)) if span.size else 1.0)
        for lag in LAGS:
            pos = tick - lag
            val = int(grid[pos]) if pos >= 0 else -1
            for i in range(5):
                feats.append(1.0 if val == i else 0.0)
            feats.append(1.0 if val < 0 else 0.0)
        pos = tick - 7
        val = int(grid[pos]) if pos >= 0 else -1
        for typ in TYPES:
            on = 1.0 if q.machine_type == typ else 0.0
            for i in range(5):
                feats.append(on if val == i else 0.0)
            feats.append(on if val < 0 else 0.0)
        same = np.zeros(5)
        anyf = np.zeros(5)
        ns = na = 0
        for other in episodes[q.episode_id]:
            if other == key:
                continue
            fr, cov = _window_fracs(grids[other], tick, 16)
            if cov == 0 and fr.sum() == 0:
                continue
            anyf += fr
            na += 1
            if types[other] == q.machine_type:
                same += fr
                ns += 1
        feats.extend((same / ns).tolist() if ns else [0.0] * 5)
        feats.append(float(ns))
        feats.extend((anyf / na).tolist() if na else [0.0] * 5)
        # streak of zeros at the end
        streak = 0
        for t in range(tick - 1, max(-1, tick - 12), -1):
            if t >= 0 and grid[t] == 0:
                streak += 1
            elif t >= 0 and grid[t] >= 0:
                break
        feats.append(streak / 12)
        return feats

    for key, q in meta.items():
        grid = grids[key]
        ticks = np.flatnonzero(grid >= 0)
        for tick in ticks:
            if tick < min_tick:
                continue
            xs.append(features(key, int(tick)))
            ys.append(int(grid[tick]))
        if labels is not None:
            tick = q.cutoff
            if tick >= min_tick:
                xs.append(features(key, tick))
                ys.append(PAY_I[labels[q.query_id]])
                qids.append(q.query_id)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int16), meta, episodes, grids


def predict_queries(model, queries, history):
    meta, episodes, grids, _ = pack(queries, history)
    # reuse feature function by a tiny local fit-less path: call build without labels
    # and only the cutoff rows. Easier to duplicate the cutoff loop.
    x, _, meta, episodes, grids = build_xy(queries, history, labels=None, min_tick=0)
    # build_xy without labels skips cutoffs. So do a dedicated cutoff matrix.
    return None


def cutoff_matrix(queries, history):
    # Build features only at cutoffs by using a fake label path with a sentinel and filtering.
    meta, episodes, grids, width = pack(queries, history)
    types = {key: q.machine_type for key, q in meta.items()}

    def features(key, tick):
        q = meta[key]
        grid = grids[key]
        feats = []
        for typ in TYPES:
            feats.append(1.0 if q.machine_type == typ else 0.0)
        for sig in (-1, 0, 1):
            feats.append(1.0 if q.signal == sig else 0.0)
        fracs = {}
        for window in WINDOWS:
            fr, cov = _window_fracs(grid, tick, window)
            fracs[window] = fr
            feats.extend(fr.tolist())
            feats.append(cov)
        feats.append(float(fracs[8][0] - fracs[64][0]))
        feats.append(float(fracs[16][0] - fracs[128][0]))
        observed = grid[:tick]
        observed = observed[observed >= 0]
        feats.append(float(np.log1p(observed.size)))
        start = max(0, tick - 40)
        span = grid[start:tick]
        feats.append(float(np.mean(span < 0)) if span.size else 1.0)
        for lag in LAGS:
            pos = tick - lag
            val = int(grid[pos]) if pos >= 0 else -1
            for i in range(5):
                feats.append(1.0 if val == i else 0.0)
            feats.append(1.0 if val < 0 else 0.0)
        pos = tick - 7
        val = int(grid[pos]) if pos >= 0 else -1
        for typ in TYPES:
            on = 1.0 if q.machine_type == typ else 0.0
            for i in range(5):
                feats.append(on if val == i else 0.0)
            feats.append(on if val < 0 else 0.0)
        same = np.zeros(5)
        anyf = np.zeros(5)
        ns = na = 0
        for other in episodes[q.episode_id]:
            if other == key:
                continue
            fr, cov = _window_fracs(grids[other], tick, 16)
            if cov == 0 and fr.sum() == 0:
                continue
            anyf += fr
            na += 1
            if types[other] == q.machine_type:
                same += fr
                ns += 1
        feats.extend((same / ns).tolist() if ns else [0.0] * 5)
        feats.append(float(ns))
        feats.extend((anyf / na).tolist() if na else [0.0] * 5)
        streak = 0
        for t in range(tick - 1, max(-1, tick - 12), -1):
            if t >= 0 and grid[t] == 0:
                streak += 1
            elif t >= 0 and grid[t] >= 0:
                break
        feats.append(streak / 12.0)
        return feats

    xs = []
    ids = []
    for key, q in meta.items():
        xs.append(features(key, q.cutoff))
        ids.append(q.query_id)
    return np.asarray(xs, dtype=np.float64), ids


def score(preds):
    data = Path("public")
    meta = round_metadata(data)
    targets = read_targets(data / "validation/targets.csv")
    groups = {r["query_id"]: r["condition"] for r in read_rows(data / "validation/conditions.csv", ("query_id", "condition"))}
    reference = read_submission(data / "validation/reference.csv")
    result = contest_score(preds, targets, groups, reference, max_points=meta["max_points"])
    bits = " ".join(f"{c[:5]}={result['by_condition'][c]['loss_ratio']:.3f}" for c in result["by_condition"])
    print(f"pts={result['technical_points']:.3f} R={result['mean_loss_ratio']:.3f} {bits}")
    return result


def _pack_preds(proba, classes, ids):
    order = [int(c) for c in classes]
    preds = {}
    for i, qid in enumerate(ids):
        row = [0.0] * 5
        for j, cls in enumerate(order):
            row[cls] = float(proba[i, j])
        total = sum(row)
        preds[qid] = tuple(v / total for v in row)
    return preds


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    train_q, train_h = load_inputs(Path("public/train"))
    labels = read_labels(Path("public/train/labels.csv"), train_q)
    print("building train matrix", flush=True)
    x, y, *_ = build_xy(train_q, train_h, labels, min_tick=96)
    print("shape", x.shape, "classes", np.bincount(y, minlength=5), flush=True)
    val_q, val_h = load_inputs(Path("public/validation"))
    print("building val matrix", flush=True)
    xv, ids = cutoff_matrix(val_q, val_h)
    print("val", xv.shape, flush=True)

    print("fit hgb", flush=True)
    hgb = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=0,
    )
    hgb.fit(x, y)
    print("hgb", end=" ", flush=True)
    score(_pack_preds(hgb.predict_proba(xv), hgb.classes_, ids))

    print("fit logit", flush=True)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    clf = LogisticRegression(C=0.5, solver="lbfgs", max_iter=120, tol=1e-3)
    clf.fit(xs, y)
    xvs = scaler.transform(xv)
    print("logit", end=" ", flush=True)
    score(_pack_preds(clf.predict_proba(xvs), clf.classes_, ids))
    payload = {
        "classes": [int(c) for c in clf.classes_],
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    Path("logit_model.json").write_text(json.dumps(payload), encoding="utf-8")
    print("saved logit_model.json", Path("logit_model.json").stat().st_size, flush=True)


if __name__ == "__main__":
    main()
