# Check-in 2 — Neural Baseline: Steam Game Recommender

**Course:** DSAN 6600 — Deep Learning  
**Project:** Steam Game Recommender  
**Status:** Trained and evaluated neural baseline  
**Dataset:** [`frankjc2022/steam-dataset`](https://huggingface.co/datasets/frankjc2022/steam-dataset) (`user_games` and `games`)

## 1. Research question and scope

**Can a neural collaborative filtering model produce better personalized top-10 game recommendations than a non-personalized popularity baseline for existing Steam users?**

The goal is to recommend games a user has not already played or owned, based on recorded play history. This is an **implicit-feedback recommendation task**, not an explicit game-rating prediction problem. A recorded user–game interaction is treated as positive when `playtime_forever > 0`; that signal indicates *engagement*, not necessarily enjoyment. Games owned with zero recorded playtime are **not** treated as negative preferences.

The model is trained and evaluated for **existing users** represented in the source data. We do not claim to predict preferences for users with no prior interactions.

## 2. Data and eligibility

The downloaded `user_games.parquet` contains **11,285,290** user–game records, representing **786 unique users** and **49,830 unique game app IDs** in interactions. Of these records, **4,488,063** have positive recorded playtime, while **6,797,227** have zero recorded playtime. The EDA notebook examines this imbalance, interaction density, popularity, missingness, and game metadata.

These 786 players have unusually large libraries: the median user owns **12,674** games and has positive playtime in **6,132** games. The resulting sample has many interactions but relatively few distinct, unusually active users. The model's evaluation is therefore **within this observed user cohort**, not a representative estimate for Steam users generally.

**Modeling eligibility:** at least **15 positive-playtime games** and at least one unowned candidate game. All **786 of 786** users met these requirements.

### Reproducible per-user holdout

There are no interaction timestamps in `user_games`, so a chronological split is not possible. Instead, using **random seed 42**, each user's distinct positive-playtime games are randomly divided into:

- **Validation:** 5 positive games per user (**3,930** interactions total).
- **Test:** 5 different positive games per user (**3,930** interactions total).
- **Training:** remaining positive games, subject to a **maximum of 1,000 sampled training positives per user**.

The notebook actually uses **741,664 training positives**; the cap omits **3,738,539** otherwise available training positives. This cap reduces computation and limits the weight of exceptionally large libraries, but it also discards preference information. The omitted positives remain in users' known histories and are **not** eligible as negatives or recommendations.

User–game positives do not overlap between the train, validation, and test splits. Source data was checked for duplicate `(steamid, appid)` records; **zero duplicates were merged** in this run.

### Negative sampling and candidate masking

For neural training, the notebook draws **two negative samples for each sampled training positive**, with new samples each epoch. Negatives are sampled from games **not recorded as owned by that user** (rather than from owned-but-unplayed titles or held-out positive games). Such negatives are *unobserved*, not confirmed dislikes.

For evaluation, each user's candidate set contains the **five target positives** for that split plus eligible unowned games. All other known owned games—including zero-playtime ownership, training positives, omitted training positives, and positives held out for the opposite split—are excluded. Both models rank **identical candidates for each user**. This prevents positive-pair overlap and avoids penalizing the model for failing to recommend already-owned games.

## 3. Models and training procedure

### Popularity baseline (non-neural)

Games are scored by the number of users with that game in the **sampled training-positive data**. Recommendations are the highest-scoring eligible games after owned-game filtering. Using the same capped training positives for the popularity and neural models makes the comparison consistent, although the cap can affect which titles rank as globally popular.

### Neural Collaborative Filtering (NCF)

A **PyTorch ID-based model** learns a 32-dimensional user embedding and a 32-dimensional game embedding. Their concatenation is passed to a **64 → 32 → 1** multilayer perceptron with ReLU activations, 0.10 dropout, and a single output logit. The model learns a nonlinear preference score from user and game IDs alone. **Genres, including the Steam label `Indie`, are not features in Check-in 2.** Game metadata is used only to display titles in the qualitative examples.

| Setting | Actual value |
|---|---|
| Framework | PyTorch (CPU in the recorded run) |
| User and game embeddings | 32 dimensions each |
| MLP layers | 64 → 32 → 1 (ReLU, dropout 0.10) |
| Trainable parameters | 1,625,985 |
| Loss | `BCEWithLogitsLoss` (binary cross-entropy) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.000001 |
| Batch size | 2,048 |
| Negative : positive ratio | 2 : 1, resampled each epoch |
| Maximum epochs | 5 |
| Checkpoint selection | Highest sampled-candidate validation NDCG@10 |
| Best checkpoint | Epoch 5 |
| Random seed | 42 |

### Training evidence

| Epoch | Training BCE loss | Validation NDCG@10¹ |
|---:|---:|---:|
| 1 | 0.2895 | 0.4686 |
| 2 | 0.2282 | 0.4845 |
| 3 | 0.2220 | 0.4988 |
| 4 | 0.2140 | 0.5259 |
| 5 | **0.2003** | **0.5524** |

¹Validation ranks **5 held-out positives + 1,000 sampled unowned games per user**; these values should not be directly compared to test values obtained using more candidates.

Training loss decreased across all five epochs and sampled-candidate validation NDCG@10 increased at every epoch. The highest validation score occurred at the last configured epoch, so **the run does not demonstrate that the model has converged** or that more epochs would necessarily improve test performance.

## 4. Evaluation design and metrics

Both baselines use exactly the same five test-positive games per user. Metrics are calculated separately per user and then **averaged across 786 users**:

- **Recall@10:** fraction of a user's five withheld positive games found in top 10.
- **NDCG@10:** rank-discounted gain for those withheld games, normalized by an ideal ranking.
- **Hit Rate@10:** fraction of users with at least one withheld positive game in top 10.
- **Catalog Coverage@10:** fraction of the **49,830 observed game IDs** appearing at least once across users' top-10 lists.

We report **two distinct evaluation protocols**. The sampled test compares 5 held-out positives with 2,000 randomly selected unowned games (**2,005 total candidates per user**). The full-catalog test compares the same 5 positives with **all eligible unowned games among the 49,830 observed IDs**, averaging **35,477.1 candidates per user**. *Full catalog* means the observed dataset's game universe, not the entirety of the live Steam store.

### A. Primary result: full eligible catalog

| Metric | Popularity | Neural Collaborative Filtering | Relative NCF improvement |
|---|---:|---:|---:|
| Recall@10 | 0.0840 | **0.1817** | +116.4% |
| NDCG@10 | 0.0763 | **0.1831** | +139.8% |
| Hit Rate@10 | 0.3206 | **0.5483** | +71.0% |
| Catalog Coverage@10 | 2.01% | **4.53%** | +125.1% |

**Interpretation:** NCF recovered **714 of 3,930** held-out positive games among top-10 recommendations (Popularity: **330 of 3,930**). Its Hit Rate@10 of **0.5483** means approximately **431 of 786 users** had at least one hit. NCF surpassed popularity under the more difficult full-catalog protocol, but its **0.1817 Recall@10** also demonstrates that most held-out positives are *not* recovered in the first ten recommendations.

### B. Secondary result: sampled candidate pool

| Metric | Popularity | Neural Collaborative Filtering |
|---|---:|---:|
| Recall@10 | 0.3509 | **0.4616** |
| NDCG@10 | 0.3265 | **0.4608** |
| Hit Rate@10 | 0.8168 | **0.9059** |
| Catalog Coverage@10 | 6.28% | **8.61%** |

All **786 users** are evaluated against **2,005 candidates each** under this protocol. NCF recovered **1,814 of 3,930** held-out positive games; popularity recovered **1,379**. This result is **not** a full-catalog measure.

The difference between evaluations is substantial: NCF Recall@10 falls from **0.4616** (sampled candidates) to **0.1817** (full catalog). Candidate-pool construction materially affects recommendation performance; sampled-negative evaluation alone would give an overly optimistic picture of real catalog ranking difficulty.

## 5. Performance by user activity (sampled-candidate test only)

Users were divided into three equal-sized groups according to their number of positive training interactions **before** the 1,000-positive cap. These numbers describe the users' original activity and do **not** measure different numbers of positives actually supplied to the model after capping.

| Original activity group | Popularity NDCG@10 | NCF NDCG@10 | NCF Recall@10 |
|---|---:|---:|---:|
| Lower activity | 0.3222 | **0.4358** | 0.4351 |
| Middle activity | 0.2850 | **0.4371** | 0.4489 |
| Higher activity | 0.3723 | **0.5097** | 0.5008 |

NCF outperformed the popularity baseline in all three groups. Higher-activity users achieved the strongest mean NCF performance in this **sampled-candidate** analysis, but this is a descriptive association, not proof that longer histories caused better predictions. The notebook does **not** separately report activity-group results for the full-catalog test.

## 6. Qualitative recommendation examples (sampled-candidate test only)

The following examples come from the notebook's **2,005-candidate sampled test**, *not* its later full-catalog run. User indices are anonymized and are **not** raw Steam IDs. A **hit** is a withheld positive-playtime game recovered within the top 10; a non-hit is not necessarily a game the user would dislike.

### Successful example — User 118: 5 of 5 held-out games recovered

- **Original training positives before cap:** 5,685.
- **Held-out games:** *Subject 13*; *Sirius*; *World's Dawn*; *Replica*; *Fly and Destroy*.
- **NCF ranks:** #1 *Subject 13* (hit), #3 *Fly and Destroy* (hit), #4 *World's Dawn* (hit), #5 *Sirius* (hit), #7 *Replica* (hit).
- **Interpretation:** All five withheld games were ranked in the top ten for this sampled candidate pool.

### Successful example — User 617: 5 of 5 held-out games recovered

- **Original training positives before cap:** 1,049.
- **Held-out games:** *3 Stars of Destiny*; *Armed with Wings: Rearmed*; *Brushwood Buddies*; *Defend Your Crypt*; *Adventure World*.
- **NCF ranks:** #1 *Brushwood Buddies* (hit), #2 *3 Stars of Destiny* (hit), #3 *Defend Your Crypt* (hit), #4 *Armed with Wings: Rearmed* (hit), #6 *Adventure World* (hit).

### Failure example — User 0: 0 of 5 held-out games recovered

- **Original training positives before cap:** 259.
- **Held-out games:** *Manhunt*; *Taekwondo Grand Prix*; *App 217860*; *App 333860*; *App 575160*. App-ID placeholders indicate titles unresolved by the available metadata lookup.
- **NCF top-10 examples:** #1 *Slay the Spire*, #2 *Pillars of Eternity*, #3 *Retrowave*, #4 *The Long Dark*.
- **Observed failure:** None of the five withheld positive games appeared in the sampled top 10.

### Failure example — User 21: 0 of 5 held-out games recovered

- **Original training positives before cap:** 1,217.
- **Held-out games:** *Beware Planet Earth*; *Without Within 2*; *Bonsai*; *Trivia Vault: Classic Rock Trivia 2*; *Intelligence: Anime girls*.
- **NCF top-10 examples:** #1 *Dungeon of the ENDLESS™*, #2 *Convoy*, #3 *SYNTHETIK: Legion Rising*, #4 *Titan Souls*.
- **Observed failure:** None of the five withheld positive games appeared in the sampled top 10 despite the larger original player history. Failures therefore cannot be attributed solely to low user activity.

The observed misses may reflect limited title-level training examples, implicit-feedback ambiguity, preference heterogeneity, popularity effects, or the training cap. The examples **do not establish which explanation is causal**.

## 7. Failure analysis and dataset limitations

1. **Small, highly active user cohort.** There are only 786 distinct users, each with unusually large game libraries. This limits conclusions about a typical Steam player, very-low-history users, or unseen users. Splitting interactions from the same users supports **existing-user recommendations**, not new-user generalization.
2. **Full-catalog ranking is difficult.** Full-catalog Recall@10 is **0.1817** and the mean eligible candidate pool is **35,477.1** items. The large performance drop relative to sampled evaluation emphasizes why candidate construction must be reported.
3. **Implicit feedback does not equal preference.** Positive playtime shows some engagement, not enjoyment, and missing ownership does not establish dislike. Sampled training negatives are therefore weak labels.
4. **Training subsampling removes information.** Only **741,664** positive training interactions were used, with **3,738,539** positives omitted under the per-user 1,000-positive cap. We have not measured how changing the cap would affect model quality.
5. **Item cold-start / sparse-item evidence.** **38 of 3,930 test-positive games (0.97%)** had **zero occurrences among sampled training-positive interactions**. An ID-only model has little positive evidence for those games. This does not establish that the game is absent from the overall raw dataset.
6. **No timestamps.** Per-user random splitting cannot reproduce the real-world task of predicting *future* play based on an earlier historical cutoff. The current split measures recovery of withheld historical played games.
7. **Catalog and metadata limitations.** The ranking universe is the **49,830 game IDs observed in user interactions**, not every Steam title. The metadata title lookup does not resolve every app ID (hence placeholders such as `App 217860`). In the EDA, the Steam label **`Indie` is not a gameplay genre**; it should be separated from gameplay-genre interpretation for future metadata-based models.
8. **Model comparison is limited.** Check-in 2 includes popularity and NCF; a standalone conventional matrix-factorization model has **not yet been trained or evaluated**. The reported improvement therefore shows a benefit over popularity, not over every simpler personalized recommender.
9. **Single experimental configuration.** Results come from one fixed seed and hyperparameter setting. No variability across random seeds or hyperparameter sensitivity analysis is reported. Model convergence is not established after five epochs.

## 8. Next steps for Check-in 3

The proposed extension is a **hybrid neural recommender** that combines the existing user–game collaborative signals with game metadata such as gameplay genres, categories, platform support, developer/publisher information, and possibly descriptions. In particular:

- Separate **gameplay genres** from classification labels such as `Indie`, `Early Access`, and `Free To Play`; check which other labels co-occur with `Indie` before making feature decisions.
- Investigate whether metadata helps titles with few or no sampled training-positive examples and improves full-catalog ranking or long-tail coverage.
- Evaluate changes to the 1,000-positives-per-user cap and consider a more demanding personalized baseline (e.g., conventional matrix factorization) if time permits.
- Keep the **same user cohort, held-out positives, candidate-filtering policy, and full-catalog metrics** for meaningful comparisons.

These are **proposed experiments**, not completed Check-in 2 results.

## 9. Reproduction and evidence

Run notebooks in order from the project root with a Python 3.12 environment and the dependencies used by the notebooks (`duckdb`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `torch`, `datasets` as needed for download):

1. `notebooks/00_download_data.ipynb` — creates `data/raw/user_games.parquet`, `data/raw/games.parquet`, and `data/raw/manifest.json`.
2. `notebooks/01_eda.ipynb` — audits the interactions and metadata.
3. `notebooks/02_neural_baseline.ipynb` — trains NCF, evaluates popularity and NCF, prints qualitative examples, and saves results. Set **`RUN_FULL_CATALOG_TEST = True`** to reproduce the full-catalog evaluation.

Saved lightweight results (commit these to Git; do not commit large raw datasets or model weights):

| Path | Contents |
|---|---|
| `results/checkin2_config.json` | Data split and training hyperparameters; confirms the full-catalog run |
| `results/checkin2_training_history.csv` | Training BCE loss and sampled-validation metrics by epoch |
| `results/checkin2_sampled_test_metrics.csv` | Both models' sampled-candidate test metrics |
| `results/checkin2_full_catalog_test_metrics.csv` | Both models' primary full-catalog metrics |
| `results/checkin2_activity_groups.csv` | Sampled-candidate metrics by original user activity |
| `models/ncf_checkin2.pt` | Local trained PyTorch checkpoint (not required in Git) |

**Evidence for the progress video:** show the completed training cell (loss + validation trajectory), the full-catalog and sampled-candidate tables, and the success/failure examples. The video should explicitly describe the 786-user dataset limitation and distinguish full-catalog from sampled-candidate scores.
