"""Command line entry point.

  python -m src.run convert --src data/train.jsonl --dst data/train.parquet
  python -m src.run convert --src data/test.jsonl  --dst data/test.parquet
  python -m src.run validate --train data/train.parquet --sample 0.05
  python -m src.run submit --train data/train.parquet --test data/test.parquet
"""
import argparse

import pandas as pd

from .covisit import build_covisit
from .data import TARGETS, jsonl_to_parquet, load, make_validation, session_histories
from .evaluate import score
from .recommend import recommend

# which co-visitation matrix serves which target
MATRIX_FOR = {"clicks": "recency", "carts": "type", "orders": "type"}


def build_matrices(df):
    return {mode: build_covisit(df, mode=mode) for mode in set(MATRIX_FOR.values())}


def predict(hist_df, matrices, popular):
    hists = session_histories(hist_df)
    return {
        t: {s: recommend(a, y, matrices[MATRIX_FOR[t]], popular) for s, (a, y) in hists.items()}
        for t in TARGETS
    }


def cmd_validate(args):
    df = load(args.train, args.sample)
    train, history, gt = make_validation(df, args.val_days)
    print(f"train events: {len(train):,} | validation history events: {len(history):,}")
    popular = train.aid.value_counts().index[:20].tolist()
    matrices = build_matrices(train)
    preds = predict(history, matrices, popular)
    per_target, joint = score(preds, gt)
    for t, v in per_target.items():
        print(f"  recall@20 {t:7s}: {v:.4f}")
    print(f"  joint score     : {joint:.4f}")


def cmd_submit(args):
    train = load(args.train, args.sample)
    test = load(args.test)
    allev = pd.concat([train, test], ignore_index=True)  # test events may be used for co-visitation
    popular = allev.aid.value_counts().index[:20].tolist()
    matrices = build_matrices(allev)
    preds = predict(test, matrices, popular)
    rows = [(f"{s}_{t}", " ".join(map(str, p))) for t in TARGETS for s, p in preds[t].items()]
    pd.DataFrame(rows, columns=["session_type", "labels"]).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows):,} rows)")


def cmd_rank(args):
    from .ranker import fit
    fit(load(args.train, args.sample), args)


def cmd_rank_submit(args):
    from .ranker import run_submit
    run_submit(load(args.train, args.sample), load(args.test), args)


def add_rank_args(p):
    p.add_argument("--train", required=True)
    p.add_argument("--sample", type=float, default=0.05)
    p.add_argument("--val-days", type=int, default=7, dest="val_days")
    p.add_argument("--max-sessions", type=int, default=40000, dest="max_sessions",
                   help="sessions used to train the rankers (more = better but needs more RAM/time)")
    p.add_argument("--n-estimators", type=int, default=300, dest="n_estimators")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    rk = sub.add_parser("rank", help="v2: train LightGBM rankers and print a validation report")
    add_rank_args(rk)

    rs = sub.add_parser("rank-submit", help="v2: train rankers, then predict the test set")
    add_rank_args(rs)
    rs.add_argument("--test", required=True)
    rs.add_argument("--out", default="submission_v2.csv.gz")
    rs.add_argument("--batch", type=int, default=50000, help="test sessions per prediction batch")

    c = sub.add_parser("convert")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)

    v = sub.add_parser("validate")
    v.add_argument("--train", required=True)
    v.add_argument("--sample", type=float, default=0.05)
    v.add_argument("--val-days", type=int, default=7, dest="val_days")

    s = sub.add_parser("submit")
    s.add_argument("--train", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--sample", type=float, default=1.0)
    s.add_argument("--out", default="submission.csv")

    args = p.parse_args()
    if args.cmd == "convert":
        jsonl_to_parquet(args.src, args.dst)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "rank":
        cmd_rank(args)
    elif args.cmd == "rank-submit":
        cmd_rank_submit(args)
    else:
        cmd_submit(args)


if __name__ == "__main__":
    main()