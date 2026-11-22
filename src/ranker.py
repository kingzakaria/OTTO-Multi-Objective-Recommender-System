"""v2: candidate generation + features + one LightGBM ranker per target.

Flow
  1. split by time: early sessions -> co-visitation tables, popularity, item2vec
                    last week      -> sessions the rankers are trained / validated on
  2. for every session, generate up to MAX_CAND candidate items and compute features
  3. label a candidate 1 if the user really did that action on it afterwards
  4. train LGBMRanker (lambdarank) for clicks, carts and orders
  5. validate (candidate recall + recall@20 vs the v1 baseline) or write a submission
"""
import csv
import gc
import gzip
import time
from collections import defaultdict

import numpy as np

from .covisit import build_covisit
from .data import TARGETS, make_validation
from .evaluate import score
from .recommend import HIST_TYPE_W, recommend

MAX_CAND = 100          # candidates per session
N_SEEN = 30             # at most this many of them come from the session itself
ITEM_SPACE = 1_900_000  # OTTO article ids are < 1.86M
MATRIX_FOR = {"clicks": "recency", "carts": "type", "orders": "type"}

FEATURES = [
    "seen_w", "n_click", "n_cart", "n_order", "since_last",     # item inside the session
    "cv_recency", "cv_type",                                    # co-visitation scores
    "sim_last", "sim_mean",                                     # item2vec similarity
    "pop_click", "pop_cart", "pop_order",                       # global popularity (log)
    "s_len", "s_carts", "s_orders", "s_hours", "last_type",     # session
]
F = len(FEATURES)
_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data helpers
def histories(df):
    """-> list of (session, aids, types, ts), events ordered oldest -> newest."""
    d = df.sort_values(["session", "ts"], kind="stable")
    s, a = d.session.to_numpy(), d.aid.to_numpy()
    ty, t = d.type.to_numpy(), d.ts.to_numpy()
    bounds = np.flatnonzero(np.diff(s)) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(s)]
    return [(int(s[i]), a[i:j].tolist(), ty[i:j].tolist(), t[i:j].tolist())
            for i, j in zip(starts, ends)]


# ----------------------------------------------------------------------------- item2vec
def train_item2vec(df, dim=32, max_sessions=300_000):
    """word2vec over sessions (item = word, session = sentence). Returns (E, idx) or None."""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        log("gensim is not installed -> item2vec features are switched off (pip install gensim)")
        return None
    ids = df.session.unique()
    if len(ids) > max_sessions:
        ids = ids[:: len(ids) // max_sessions + 1]
    sub = df[df.session.isin(ids)]
    sentences = [[str(x) for x in h[1][:200]] for h in histories(sub) if len(h[1]) >= 2]
    log(f"training word2vec on {len(sentences):,} sessions")
    m = Word2Vec(sentences, vector_size=dim, window=5, min_count=2, sg=1, negative=5,
                 epochs=3, workers=4, seed=1)
    keys = np.array([int(k) for k in m.wv.index_to_key], dtype=np.int64)
    vec = m.wv.vectors.astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    n = len(keys)
    E = np.vstack([vec, np.zeros((1, dim), np.float32)])   # last row = unknown item
    idx = np.full(ITEM_SPACE, n, np.int32)
    idx[keys] = np.arange(n)
    return E, idx


def no_embeddings(dim=32):
    return np.zeros((1, dim), np.float32), np.zeros(ITEM_SPACE, np.int32)


class Resources:
    """Everything candidate generation needs: co-visitation, popularity, embeddings."""


def build_resources(df, emb):
    r = Resources()
    log("building co-visitation matrices")
    r.cv = {m: build_covisit(df, mode=m) for m in ("recency", "type")}
    cnt = df.groupby(["aid", "type"]).size()
    pop = np.zeros((ITEM_SPACE, 3), np.float32)
    pop[cnt.index.get_level_values(0).to_numpy(), cnt.index.get_level_values(1).to_numpy()] = \
        np.log1p(cnt.to_numpy())
    r.pop = pop
    r.popular = df.aid.value_counts().index[:MAX_CAND].tolist()
    r.E, r.emb_idx = emb
    return r


# ----------------------------------------------------------------------------- candidates + features
def gen_session(aids, types, ts, r):
    n = len(aids)
    tail, ttail = aids[-50:], types[-50:]
    m = len(tail)
    rec_w = np.linspace(0.1, 1.0, m)

    seen, last = defaultdict(float), {}
    for i, (a, t) in enumerate(zip(tail, ttail)):
        seen[a] += rec_w[i] * HIST_TYPE_W[t]
        last[a] = m - 1 - i                      # steps since the latest touch (0 = last event)
    cnts = {}
    for a, t in zip(aids, types):                # counts over the whole session
        c = cnts.get(a)
        if c is None:
            c = cnts[a] = [0, 0, 0]
        c[t] += 1

    r_sc, t_sc = defaultdict(float), defaultdict(float)
    recent = tail[-20:]
    for a, wi in zip(recent, np.linspace(0.1, 1.0, len(recent))):
        for b, c in r.cv["recency"].get(a, ()):
            r_sc[b] += c * wi
        for b, c in r.cv["type"].get(a, ()):
            t_sc[b] += c * wi

    cand = sorted(seen, key=seen.get, reverse=True)[:N_SEEN]
    have = set(cand)
    if r_sc or t_sc:
        mr = max(r_sc.values(), default=1.0) or 1.0
        mt = max(t_sc.values(), default=1.0) or 1.0
        pool = set(r_sc) | set(t_sc)
        for b in sorted(pool, key=lambda x: r_sc.get(x, 0.0) / mr + t_sc.get(x, 0.0) / mt, reverse=True):
            if len(cand) >= MAX_CAND:
                break
            if b not in have:
                cand.append(b)
                have.add(b)
    for b in r.popular:                           # make sure we always have enough candidates
        if len(cand) >= 30:
            break
        if b not in have:
            cand.append(b)
            have.add(b)

    k = len(cand)
    X = np.zeros((k, F), np.float32)
    X[:, 0] = [seen.get(a, 0.0) for a in cand]
    for j in range(3):
        X[:, 1 + j] = [cnts[a][j] if a in cnts else 0 for a in cand]
    X[:, 4] = [last.get(a, 99) for a in cand]
    X[:, 5] = [r_sc.get(a, 0.0) for a in cand]
    X[:, 6] = [t_sc.get(a, 0.0) for a in cand]
    C = r.E[r.emb_idx[cand]]
    lv = r.E[r.emb_idx[tail[-1]]]
    mv = r.E[r.emb_idx[tail[-10:]]].mean(0)
    mv = mv / (np.linalg.norm(mv) + 1e-8)
    X[:, 7] = C @ lv
    X[:, 8] = C @ mv
    X[:, 9:12] = r.pop[cand]
    X[:, 12] = n
    X[:, 13] = types.count(1)
    X[:, 14] = types.count(2)
    X[:, 15] = (ts[-1] - ts[0]) / 3600
    X[:, 16] = types[-1]
    return np.asarray(cand, np.int32), X


def gen_batch(hists, r, every=20_000):
    group_sids, sizes, aid_list, x_list = [], [], [], []
    for i, (sid, aids, types, ts) in enumerate(hists):
        ca, X = gen_session(aids, types, ts, r)
        group_sids.append(sid)
        sizes.append(len(ca))
        aid_list.append(ca)
        x_list.append(X)
        if every and (i + 1) % every == 0:
            log(f"  candidates for {i + 1:,}/{len(hists):,} sessions")
    return group_sids, np.array(sizes), np.concatenate(aid_list), np.vstack(x_list)


def make_labels(group_sids, sizes, aids, truth):
    y = np.zeros(len(aids), np.int8)
    pos = 0
    for sid, k in zip(group_sids, sizes):
        t = truth.get(sid)
        if t:
            y[pos:pos + k] = np.isin(aids[pos:pos + k], list(t))
        pos += k
    return y


def positive_groups(X, y, sizes):
    """Keep only sessions with at least one correct candidate (the rest teach nothing)."""
    starts = np.cumsum(sizes) - sizes
    has = np.add.reduceat(y.astype(np.int32), starts) > 0
    mask = np.repeat(has, sizes)
    return X[mask], y[mask], sizes[has]


# ----------------------------------------------------------------------------- model
def fit_ranker(name, train, valid, n_estimators):
    import warnings

    import lightgbm as lgb
    warnings.filterwarnings("ignore", category=lgb.sklearn.LGBMDeprecationWarning)
    Xt, yt, gt = positive_groups(*train)
    Xv, yv, gv = positive_groups(*valid)
    if len(gt) == 0 or len(gv) == 0:
        raise SystemExit(f"no positive examples for '{name}': use more sessions (--max-sessions / --sample)")
    log(f"training LightGBM '{name}': {len(gt):,} sessions, {len(yt):,} rows, {int(yt.sum()):,} positives")
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=n_estimators, learning_rate=0.05,
                           num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    model.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[20],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    log(f"  '{name}' best iteration: {model.best_iteration_ or n_estimators}")
    return model


def top20(model, group_sids, sizes, aids, X):
    s = model.predict(X)
    out, pos = {}, 0
    for sid, k in zip(group_sids, sizes):
        order = np.argsort(-s[pos:pos + k])[:20]
        out[sid] = aids[pos:pos + k][order].tolist()
        pos += k
    return out


def candidate_recall(group_sids, sizes, aids, truth):
    hits = total = pos = 0
    for sid, k in zip(group_sids, sizes):
        t = truth.get(sid)
        if t:
            hits += len(set(aids[pos:pos + k].tolist()) & t)
            total += len(t)
        pos += k
    return hits / total if total else 0.0


# ----------------------------------------------------------------------------- orchestration
def fit(df, a):
    """Train the three rankers. Prints a validation report. Returns (models, embeddings)."""
    train, history, gt = make_validation(df, a.val_days)
    log(f"base events: {len(train):,} | validation-period events: {len(history):,}")

    hs = histories(history)
    np.random.default_rng(0).shuffle(hs)
    val_all = [h for h in hs if h[0] % 5 == 0]
    tr = [h for h in hs if h[0] % 5 != 0][: a.max_sessions]
    va = val_all[: max(5000, a.max_sessions // 4)]
    log(f"ranker sessions: {len(tr):,} train / {len(va):,} validation")

    emb = train_item2vec(train) or no_embeddings()
    r = build_resources(train, emb)

    log("generating candidates (train sessions)")
    tr_sids, tr_sz, tr_aids, tr_X = gen_batch(tr, r)
    log("generating candidates (validation sessions)")
    va_sids, va_sz, va_aids, va_X = gen_batch(va, r)

    va_set = set(va_sids)
    gt_val = {t: {s: v for s, v in gt[t].items() if s in va_set} for t in TARGETS}

    models, preds = {}, {}
    for t in TARGETS:
        ytr = make_labels(tr_sids, tr_sz, tr_aids, gt[t])
        yva = make_labels(va_sids, va_sz, va_aids, gt[t])
        models[t] = fit_ranker(t, (tr_X, ytr, tr_sz), (va_X, yva, va_sz), a.n_estimators)
        preds[t] = top20(models[t], va_sids, va_sz, va_aids, va_X)

    log("scoring the v1 baseline on the same validation sessions")
    base = {t: {h[0]: recommend(h[1], h[2], r.cv[MATRIX_FOR[t]], r.popular) for h in va} for t in TARGETS}

    cand_rec = {t: candidate_recall(va_sids, va_sz, va_aids, gt_val[t]) for t in TARGETS}
    per_base, joint_base = score(base, gt_val)
    per_v2, joint_v2 = score(preds, gt_val)

    print("\n=== validation report ===")
    print(f"{'target':8s} {'cand. recall':>13s} {'baseline R@20':>14s} {'v2 R@20':>9s}")
    for t in TARGETS:
        print(f"{t:8s} {cand_rec[t]:13.4f} {per_base[t]:14.4f} {per_v2[t]:9.4f}")
    print(f"{'JOINT':8s} {'':13s} {joint_base:14.4f} {joint_v2:9.4f}\n")

    imp = sorted(zip(FEATURES, models["orders"].feature_importances_), key=lambda x: -x[1])
    print("top features for orders:", ", ".join(f"{n}({v})" for n, v in imp[:6]), "\n")
    return models, emb


def run_submit(train_df, test_df, a):
    import pandas as pd
    models, emb = fit(train_df, a)

    log("building resources from all events (train sample + test)")
    allev = pd.concat([train_df, test_df], ignore_index=True)
    gc.collect()
    r = build_resources(allev, emb)
    del allev
    gc.collect()

    test_h = histories(test_df)
    log(f"predicting for {len(test_h):,} test sessions")
    opener = gzip.open if a.out.endswith(".gz") else open
    rows = 0
    with opener(a.out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_type", "labels"])
        for i in range(0, len(test_h), a.batch):
            sids, sz, aids, X = gen_batch(test_h[i:i + a.batch], r, every=0)
            for t in TARGETS:
                for sid, labels in top20(models[t], sids, sz, aids, X).items():
                    w.writerow([f"{sid}_{t}", " ".join(map(str, labels))])
                    rows += 1
            log(f"  {min(i + a.batch, len(test_h)):,}/{len(test_h):,} sessions done")
    log(f"wrote {a.out} ({rows:,} rows)")
