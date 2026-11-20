"""Joint recall@20 metric used by the OTTO competition."""

WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}


def recall_at_20(preds, truth_by_session):
    """preds: {session: [aid, ...]}, truth_by_session: {session: set(aid)}."""
    hits = total = 0
    for s, truth in truth_by_session.items():
        hits += len(set(preds.get(s, [])[:20]) & truth)
        total += min(len(truth), 20)
    return hits / total if total else 0.0


def score(preds_by_target, gt_by_target):
    per_target = {t: recall_at_20(preds_by_target[t], gt_by_target[t]) for t in WEIGHTS}
    joint = sum(WEIGHTS[t] * per_target[t] for t in WEIGHTS)
    return per_target, joint
