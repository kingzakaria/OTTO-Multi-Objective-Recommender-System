# OTTO – Multi-Objective Recommender System

A session-based recommender for e-commerce, built for the Kaggle competition
[OTTO – Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system).

**Current best result (v1b, co-visitation baseline, 20% of train): Kaggle public score 0.56156 / private score 0.56189**
(late submission, after the competition deadline).

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
- About 12M anonymized sessions, 220M events and 1.8M distinct articles. There are no text or image features, so this is a collaborative-filtering problem.
- `train.jsonl` holds full sessions. `test.jsonl` holds sessions cut at a point in time (about 1.67M sessions).
- Each event looks like `{"aid": 1517085, "ts": 1659304800025, "type": "clicks"}`.

### Metric
Recall@20 per target, combined into one score:

```
score = 0.10 * recall_clicks + 0.30 * recall_carts + 0.60 * recall_orders
recall = sum(hits) / sum(min(20, number of ground-truth items))
```

Orders carry 60% of the score.

## 2. Results

### Kaggle leaderboard (late submission)

| Version | Method | Public score | Private score |
|---------|--------|--------------|---------------|
| v1 | Co-visitation baseline (no ML model), 10% of train sessions | 0.55687 | 0.55776 |
| v1b | Same method, 20% of train sessions | **0.56156** | **0.56189** |

Doubling the training sample gave about +0.005, so more data alone helps only a little. The next gain has to come from a learned ranker (see the roadmap).

### Local validation of v1 (5% sample, last 7 days held out)

| Target | Recall@20 | Weight |
|--------|-----------|--------|
| Clicks | 0.4246 | 10% |
| Carts  | 0.3557 | 30% |
| Orders | 0.6188 | 60% |
| **Joint** | **0.5204** | |

The local number is only an approximation: it uses a 5% sample and my own split, so it differs from the leaderboard score.

## 3. Method (v1: co-visitation baseline)

The baseline contains no trained model. It counts and sorts:

1. **Session items first.** The items the user already interacted with, ranked by recency and event type (clicks 1, carts 6, orders 3).
2. **Co-visitation neighbours.** For the recent items of the session, add items that often appear in the same sessions. Two matrices are used:
   - `recency`-weighted, used for clicks.
   - `type`-weighted (carts and orders count more than clicks), used for carts and orders.
   - Pairs must occur within 24 hours, only the last 30 events per session are used, and the top 20 neighbours are kept per item.
3. **Popular items** fill any remaining slots.

### Validation split
The last 7 days of train become validation sessions. Each is cut at a random point. The history before the cut is the input, and the events after it are the ground truth. The co-visitation tables are built only from earlier sessions, so there is no leakage.

## 4. Project structure

| File               | Role                                                                       |
|--------------------|----------------------------------------------------------------------------|
| `src/data.py`      | jsonl → parquet conversion, session sampling, time-based validation split  |
| `src/covisit.py`   | co-visitation matrices (`recency` and `type` weighted)                     |
| `src/recommend.py` | baseline: session items → co-visitation neighbours → popular items         |
| `src/evaluate.py`  | the joint recall@20 metric                                                 |
| `src/run.py`       | command line: `convert`, `validate`, `submit`                              |

## 5. How to run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put `train.jsonl` and `test.jsonl` from the Kaggle competition data into the `data/` folder (they are ignored by Git), then:

```powershell
# 1. convert to parquet
python -m src.run convert --src data/train.jsonl --dst data/train.parquet
python -m src.run convert --src data/test.jsonl  --dst data/test.parquet

# 2. local validation on 5% of the sessions
python -m src.run validate --train data/train.parquet --sample 0.05

# 3. create the submission (20% of train sessions, all test sessions)
python -m src.run submit --train data/train.parquet --test data/test.parquet --sample 0.2 --out submission.csv.gz
```

The submission file must have 5,015,409 rows plus a header. A `.csv.gz` file is accepted by Kaggle, so it can be uploaded directly.
You can delete the large `.jsonl` files after conversion to save disk space (about 11 GB).

Notes:
- `submit` prints nothing until it finishes, and takes a long time because it predicts for about 1.67M sessions.
- If memory runs out, lower `--sample`.

## 6. Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 0. Setup | Environment, data, parquet conversion | Done |
| 1. Baseline | Co-visitation pipeline, validation, first submissions | Done (public 0.55687 → 0.56156 with 20% of train) |
| 2. Analysis | Notebook on session lengths, event mix, item popularity | To do |
| 3. Better candidates | item2vec (gensim), tuned co-visitation, buy→buy matrix; measure candidate recall | To do |
| 4. Ranker | Features + LightGBM ranker, one model per target, orders first | To do |
| 5. Scale up | More training sessions, tuning | To do |

### Planned ranker (v2)
Candidates from several sources are scored by **LightGBM** (`LGBMRanker`, one model per target). Training examples come from sessions in the last week, with candidates labelled 1 if they appear in the ground truth. Planned features:
- **Session:** length, number of clicks/carts/orders, time since the last event.
- **Item:** global click, cart and order counts and their ratios.
- **Session × item:** already seen or carted, co-visitation score and rank, embedding similarity to the last item.

Sequence models (GRU or Transformer) are an optional later addition as an extra candidate source.

## 7. Limitations
- v1 has no learned model, so it cannot weigh signals against each other.
- Only 10% to 20% of the training sessions are used for the submissions because of memory limits.
- The leaderboard score is from a late submission, so it is not an official ranking.

## 8. References
- [OTTO Recommender Systems Dataset](https://github.com/otto-de/recsys-dataset)
- [3rd place solution (bschifferer/Kaggle-Otto-Comp)](https://github.com/bschifferer/Kaggle-Otto-Comp)
- [Bronze-medal solution (ajisamudra)](https://github.com/ajisamudra/kaggle-otto-multi-objective-recsys)