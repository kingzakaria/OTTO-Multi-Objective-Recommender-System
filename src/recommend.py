"""Baseline recommender: recent items first, then co-visitation neighbours, then popular items."""
from collections import defaultdict

import numpy as np

HIST_TYPE_W = {0: 1.0, 1: 6.0, 2: 3.0}


def recommend(aids, types, cv, popular, k=20, tail=20):
    """aids/types are the session so far, oldest -> newest. Returns up to k aids."""
    # 1) items already seen in the session, weighted by recency and event type
    w = np.linspace(0.1, 1.0, len(aids))
    seen = defaultdict(float)
    for a, t, wi in zip(aids, types, w):
        seen[a] += wi * HIST_TYPE_W[t]
    ranked_seen = sorted(seen, key=seen.get, reverse=True)
    if len(ranked_seen) >= k:
        return ranked_seen[:k]

    # 2) neighbours of the most recent items from the co-visitation matrix
    scores = defaultdict(float)
    hist = aids[-tail:]
    hw = np.linspace(0.1, 1.0, len(hist))
    for a, wi in zip(hist, hw):
        for b, c in cv.get(a, ()):
            scores[b] += c * wi
    out = list(ranked_seen)
    present = set(out)
    for b in sorted(scores, key=scores.get, reverse=True):
        if len(out) >= k:
            break
        if b not in present:
            out.append(b)
            present.add(b)

    # 3) fall back to globally popular items
    for b in popular:
        if len(out) >= k:
            break
        if b not in present:
            out.append(b)
            present.add(b)
    return out[:k]
