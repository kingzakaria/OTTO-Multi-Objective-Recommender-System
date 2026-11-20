"""Co-visitation matrices: which items tend to appear together in the same session."""
from collections import defaultdict

import numpy as np
import pandas as pd

# weight given to the *neighbour* item depending on how the user interacted with it
TYPE_W = np.array([1.0, 6.0, 3.0])  # clicks, carts, orders


def build_covisit(df, mode="recency", topk=20, window=24 * 3600, tail=30, chunks=50):
    """Return {aid: [(neighbour_aid, weight), ...]} with the top-k neighbours per item.

    mode="recency": neighbour weight grows for more recent events (good for clicks)
    mode="type":    neighbour weight depends on its event type (good for carts / orders)

    Only the last `tail` events of each session are used, and only pairs within `window`
    seconds of each other. The self-merge is done in chunks of sessions to bound memory.
    """
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df = df[df.groupby("session").cumcount() < tail].copy()
    ts_min, ts_max = int(df.ts.min()), int(df.ts.max())
    df["chunk"] = df.session % chunks

    acc = None
    for _, d in df.groupby("chunk"):
        m = d.merge(d, on="session", suffixes=("_x", "_y"))
        m = m[(m.aid_x != m.aid_y) & ((m.ts_x - m.ts_y).abs() < window)]
        if mode == "recency":
            m["wgt"] = 1 + 3 * (m.ts_x - ts_min) / (ts_max - ts_min + 1)
        else:
            m["wgt"] = TYPE_W[m.type_y.values]
        agg = m.groupby(["aid_x", "aid_y"]).wgt.sum()
        acc = agg if acc is None else acc.add(agg, fill_value=0)

    top = acc.reset_index().sort_values(["aid_x", "wgt"], ascending=[True, False])
    top = top.groupby("aid_x").head(topk)
    cv = defaultdict(list)
    for x, y, w in zip(top.aid_x.values, top.aid_y.values, top.wgt.values):
        cv[int(x)].append((int(y), float(w)))
    return cv
