"""Data loading, parquet conversion and time-based validation split."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TYPE2ID = {"clicks": 0, "carts": 1, "orders": 2}
ID2TYPE = {v: k for k, v in TYPE2ID.items()}
TARGETS = ["clicks", "carts", "orders"]


def _to_table(s, a, t, y):
    return pa.table({
        "session": pa.array(s, pa.int32()),
        "aid": pa.array(a, pa.int32()),
        "ts": pa.array(t, pa.int32()),  # seconds
        "type": pa.array(y, pa.int8()),
    })


def jsonl_to_parquet(src, dst, chunk_sessions=200_000):
    """Stream the Kaggle .jsonl file into a compact parquet file (one row per event)."""
    writer, s, a, t, y, n = None, [], [], [], [], 0
    with open(src) as f:
        for line in f:
            rec = json.loads(line)
            for e in rec["events"]:
                s.append(rec["session"])
                a.append(e["aid"])
                t.append(e["ts"] // 1000)
                y.append(TYPE2ID[e["type"]])
            n += 1
            if n % chunk_sessions == 0:
                tbl = _to_table(s, a, t, y)
                writer = writer or pq.ParquetWriter(dst, tbl.schema)
                writer.write_table(tbl)
                s, a, t, y = [], [], [], []
    if s:
        tbl = _to_table(s, a, t, y)
        writer = writer or pq.ParquetWriter(dst, tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()
    print(f"converted {n:,} sessions -> {dst}")


def load(path, sample=1.0):
    """Load parquet. `sample` keeps a deterministic fraction of sessions (for laptops)."""
    df = pd.read_parquet(path)
    if sample < 1.0:
        df = df[(df.session % 1000) < int(sample * 1000)]
    return df.reset_index(drop=True)


def make_validation(df, val_days=7, seed=42):
    """Mimic the competition: last `val_days` of train become validation sessions,
    each truncated at a random point. Returns (train, history, ground_truth).

    ground_truth[target][session] = set of aids
      clicks -> the first click after the cut; carts/orders -> all aids after the cut.
    """
    rng = np.random.default_rng(seed)
    cutoff = df.ts.max() - val_days * 86400
    g = df.groupby("session").ts.agg(["min", "max"])
    train = df[df.session.isin(g.index[g["max"] < cutoff])]
    val = df[df.session.isin(g.index[g["min"] >= cutoff])]
    val = val.sort_values(["session", "ts"], kind="stable").reset_index(drop=True)

    sizes = val.groupby("session").size()
    sizes = sizes[sizes >= 2]
    cuts = pd.Series(rng.integers(1, sizes.values), index=sizes.index)
    val = val[val.session.isin(sizes.index)].copy()
    val["pos"] = val.groupby("session").cumcount()
    val["cut"] = val.session.map(cuts)

    history = val[val.pos < val.cut]
    future = val[val.pos >= val.cut]

    gt = {
        "clicks": future[future.type == 0].groupby("session").aid.first().map(lambda x: {x}).to_dict(),
        "carts": future[future.type == 1].groupby("session").aid.agg(set).to_dict(),
        "orders": future[future.type == 2].groupby("session").aid.agg(set).to_dict(),
    }
    return train, history[["session", "aid", "ts", "type"]], gt


def session_histories(df):
    """-> {session: (aids oldest->newest, types)}"""
    d = df.sort_values(["session", "ts"], kind="stable")
    g = d.groupby("session").agg(aids=("aid", list), types=("type", list))
    return dict(zip(g.index, zip(g.aids, g.types)))
