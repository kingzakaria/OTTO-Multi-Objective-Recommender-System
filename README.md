# OTTO – Multi-Objective Recommender System

A session-based recommender for e-commerce, built on the data from the Kaggle competition
[OTTO – Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system).

## 1. Description

OTTO is the largest German online shop. Given the events of a user session so far
(clicks, cart additions, orders on articles), the system predicts what the user will do next.

**Multi-objective** means three predictions per session, each a list of up to **20 article ids (aids)**:

| Target   | What must be predicted                                  |
|----------|---------------------------------------------------------|
| `clicks` | the next article the user clicks                        |
| `carts`  | the articles the user will add to the cart afterwards   |
| `orders` | the articles the user will order afterwards             |

### Data
- ~12M real, anonymized sessions, ~220M events, ~1.8M distinct articles. No text or image features, so it is a collaborative-filtering problem.
- `train.jsonl` holds full sessions. `test.jsonl` holds sessions cut at a point in time.
- Each event: `{"aid": 1517085, "ts": 1659304800025, "type": "clicks"}`.
- Download (accept the competition rules on Kaggle first):
  `kaggle competitions download -c otto-recommender-system`

### Metric
Recall@20 per target, combined into one score:

```
score = 0.10 * recall_clicks + 0.30 * recall_carts + 0.60 * recall_orders
recall = sum(hits) / sum(min(20, number of ground-truth items))
```
Orders carry 60% of the score, so predicting purchases is the priority.

## 2. Idea / architecture

A two-stage recommender, the standard approach for this kind of problem:

```
session history ──► 1. CANDIDATE GENERATION ──► ~50–200 candidates
                       - items already seen in the session
                       - co-visitation neighbours (recency / event-type weighted)
                       - item2vec (word2vec) neighbours
                       - popular items
                                  │
                                  ▼
                    2. FEATURE ENGINEERING
                       session, item and session×item features
                                  │
                                  ▼
                    3. RANKER (LightGBM, one model per target)
                                  │
                                  ▼
                       top-20 per target ──► submission.csv
```

Why it works: scoring 1.8M articles per session is impossible, so cheap statistics narrow the
list, and a learned model orders the short list.

## 3. What is already in this project

A working baseline (no ML model yet) so you have a score to beat:

| File               | Role                                                                       |
|--------------------|----------------------------------------------------------------------------|
| `src/data.py`      | jsonl→parquet conversion, session sampling, time-based validation split    |
| `src/covisit.py`   | co-visitation matrices (`recency` and `type` weighted)                     |
| `src/recommend.py` | baseline: session items → co-visitation neighbours → popular items         |
| `src/evaluate.py`  | the joint recall@20 metric                                                 |
| `src/run.py`       | CLI: `convert`, `validate`, `submit`                                       |

The code was smoke-tested on a small synthetic dataset. It has **not** been run on the real data,
so check memory use and runtime on a sample first.

## 4. How to run

```bash
pip install -r requirements.txt
kaggle competitions download -c otto-recommender-system -p data && unzip data/otto-recommender-system.zip -d data

python -m src.run convert --src data/train.jsonl --dst data/train.parquet
python -m src.run convert --src data/test.jsonl  --dst data/test.parquet

# local validation on 5% of sessions (start small!)
python -m src.run validate --train data/train.parquet --sample 0.05

# create the submission file
python -m src.run submit --train data/train.parquet --test data/test.parquet
```

Validation mimics the competition: the last 7 days of train become validation sessions, each
cut at a random point; the held-out future events are the ground truth.
Memory note: the co-visitation self-merge is the heavy step. Lower `tail`, raise `chunks`,
or lower `--sample` if you run out of RAM.

## 5. Roadmap (what to build)

| Phase | Goal | Done when |
|-------|------|-----------|
| 0. Setup | Environment, Kaggle API, data downloaded | `convert` produces parquet files |
| 1. Baseline | Run the included pipeline on a 5% sample | You have a validation score for clicks / carts / orders |
| 2. Analysis | Notebook: session lengths, event mix, item popularity, time patterns | Short list of findings that guide the features |
| 3. Better candidates | Add item2vec (gensim), tune co-visitation (window, tail, top-k, weights), add a buy→buy matrix | Candidate recall (is the true item in the list?) goes up |
| 4. Ranker | Features + LightGBM per target (binary: was this candidate the next item?) | Validation score beats the Phase 1 baseline |
| 5. Scale up | Use more sessions, add test events to co-visitation, tune on full data | Stable score with more data |
| 6. Submission and report | Create `submission.csv`, submit to Kaggle, write up results | Public leaderboard score + short report |

### Feature ideas for the ranker (Phase 4)
- **Session:** length, number of clicks/carts/orders, time since last event, duration.
- **Item:** global click/cart/order counts, cart-to-click and order-to-cart ratios, recent popularity.
- **Session × item:** was it already seen / carted, position and time since last interaction, co-visitation score and rank per matrix, embedding similarity to the last item.

### Ideas after the core pipeline
- Sequence models (GRU or Transformer) on the session as an extra candidate source or feature.
- Tune the number of candidates against recall and runtime.
- Ensemble several candidate generators and rankers.

## 6. Deliverables
1. The code repository (this folder).
2. An analysis notebook (Phase 2).
3. A validation table: recall@20 for clicks, carts, orders and the joint score per phase.
4. `submission.csv` and the Kaggle leaderboard score.
5. A short report: idea, method, results, what worked and what did not.

## 7. References
- [OTTO Recommender Systems Dataset](https://github.com/otto-de/recsys-dataset)
- [3rd place solution (bschifferer/Kaggle-Otto-Comp)](https://github.com/bschifferer/Kaggle-Otto-Comp)
- [Bronze-medal solution (ajisamudra)](https://github.com/ajisamudra/kaggle-otto-multi-objective-recsys)
- [LightGBM multi-objective paper](https://unpaywall.org/10.1145%2F3627377.3627387)
