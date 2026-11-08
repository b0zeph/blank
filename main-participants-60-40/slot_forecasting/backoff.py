"""Hierarchical payout-distribution forecaster.

This model keeps the participant-facing fit/predict interface used by the
experiment runner.  It learns payout distributions from training queries and
their observed history, then backs off from detailed contexts to broader
contexts when a context is sparse.
"""
from collections import defaultdict

from .schema import PAYOUTS, index_history


PAY_I = {payout: i for i, payout in enumerate(PAYOUTS)}


class BackoffForecaster:
    def __init__(self, strengths=(40, 30, 20, 12, 8, 12), prior=0.2):
        self.strengths = tuple(float(x) for x in strengths)
        if len(self.strengths) != 6 or any(x <= 0 for x in self.strengths):
            raise ValueError("strengths must contain six positive values")
        self.prior = float(prior)
        if self.prior <= 0:
            raise ValueError("prior must be positive")
        self.levels = None

    @staticmethod
    def _features(sequence, tick, machine_type, signal):
        recent = [sequence[t] for t in range(tick - 16, tick) if t in sequence]
        long = [sequence[t] for t in range(tick - 64, tick) if t in sequence]
        n = len(recent)
        ln = len(long)
        z = sum(p == 0 for p in recent) / n if n else 0.5
        zl = sum(p == 0 for p in long) / ln if ln else z
        return {
            "type": machine_type,
            "signal": signal,
            "z": min(4, int(z * 5)),
            "zl": min(4, int(zl * 5)),
            "lag1": sequence.get(tick - 1, -1),
            "lag7": sequence.get(tick - 7, -1),
            "lag8": sequence.get(tick - 8, -1),
            "short": int(n < 12),
        }

    @staticmethod
    def _keys(f):
        return (
            ("all",),
            ("t", f["type"]),
            ("tz", f["type"], f["z"]),
            ("tz7", f["type"], f["z"], f["lag7"]),
            ("tz17", f["type"], f["z"], f["lag7"], f["lag1"]),
            ("tzs7", f["type"], f["signal"], f["z"], f["lag7"]),
        )

    @staticmethod
    def _add(store, key, payout):
        store[key][PAY_I[payout]] += 1

    def fit(self, queries, history, labels=None):
        if labels is None:
            raise ValueError("BackoffForecaster requires training labels")
        queries = tuple(queries)
        indexed = index_history(queries, history)
        self.levels = [defaultdict(lambda: [0] * len(PAYOUTS)) for _ in range(6)]

        for query in queries:
            rows = indexed.get((query.episode_id, query.machine_id), [])
            sequence = {row.tick: row.payout for row in rows}
            events = [(row.tick, row.payout) for row in rows]
            events.append((query.cutoff, labels[query.query_id]))

            for tick, payout in events:
                f = self._features(sequence, tick, query.machine_type, query.signal)
                for level, key in zip(self.levels, self._keys(f)):
                    self._add(level, key, payout)

        return self

    def _predict_one(self, sequence, query):
        f = self._features(sequence, query.cutoff, query.machine_type, query.signal)
        keys = self._keys(f)
        dist = [1.0 / len(PAYOUTS)] * len(PAYOUTS)

        for level, key, strength in zip(self.levels, keys, self.strengths):
            row = level.get(key)
            if not row:
                continue
            n = sum(row)
            if n <= 0:
                continue
            dist = [(strength * old + count) / (strength + n)
                    for old, count in zip(dist, row)]

        total = sum(dist)
        return tuple(value / total for value in dist)

    def predict(self, queries, history):
        if self.levels is None:
            raise ValueError("fit must be called before predict")
        queries = tuple(queries)
        indexed = index_history(queries, history)
        predictions = {}
        for query in queries:
            rows = indexed.get((query.episode_id, query.machine_id), [])
            sequence = {row.tick: row.payout for row in rows}
            predictions[query.query_id] = self._predict_one(sequence, query)
        return predictions
