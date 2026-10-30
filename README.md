DSAN 6600 — Deep Learning Final Project: Steam Game Recommender

Project Overview:
This project investigates whether a neural recommendation system can learn useful player preferences from Steam user–game interaction data and generate personalized top-N game recommendations.
The central research question is:
Can a neural collaborative filtering model produce better personalized Steam game recommendations than simpler recommendation baselines?

Check-in 1 establishes the problem, dataset, exploratory data analysis, and evaluation protocol. 
Check-in 2 will implement a neural recommender and compare it with simpler baselines. 
Check-in 3 will extend the neural model with additional game information.

Motivation:
Digital game platforms contain catalogs that are far larger than any individual player can realistically explore. Recommendation systems help users discover games that align with their interests while also helping platforms surface relevant content from a large catalog.
Steam provides a useful setting for studying recommendation because user libraries contain implicit behavioral signals such as game ownership and lifetime playtime. At the same time, the data presents realistic recommender-system challenges including sparse user–item interactions, popularity imbalance, low-history users, long-tail games, and ambiguous implicit feedback.
This project is also intentionally scoped to emphasize reasoning, evaluation, and failure analysis rather than simply building the most complex neural architecture.

Problem Definition:
The project is an implicit-feedback top-N recommendation problem.
For a Steam user with a history of game interactions, the system will rank candidate games according to predicted user preference. The final output for a user is a ranked list of recommended games.
The project will initially treat a recorded interaction with positive lifetime playtime as evidence of an observed positive preference signal:
positive interaction = playtime_forever > 0
This definition may be refined based on the exploratory analysis of playtime thresholds. An important limitation is that an observed interaction does not necessarily mean that the player liked the game, while an unobserved interaction does not necessarily mean that the player dislikes it.

Success Criteria:
The main success criterion is whether the neural model improves personalized ranking performance over simpler recommendation baselines under the same evaluation protocol.
The primary comparison will use:
- Recall@10
- NDCG@10
Secondary measures will include:
- Hit Rate@10
- Catalog coverage

The project will compare the neural model with at least:
1. a popularity-based recommender;
2. a traditional collaborative-filtering / matrix-factorization baseline; and
3. Neural Collaborative Filtering.
The goal is not to assume that a neural network will win, but to determine empirically whether the added model complexity produces meaningful improvements and where each approach succeeds or fails.

Semester Scope:
The project is intentionally divided into three modeling stages.
Check-in 1 — Data and Evaluation Foundation
- establish the recommendation task;
- obtain and document the Steam dataset;
- audit the user–game interaction data;
- analyze playtime, user activity, game popularity, sparsity, missingness, and metadata;
- identify likely biases and failure modes;
- establish the train/validation/test and ranking-metric protocol.

Check-in 2 — Neural Baseline
Implement Neural Collaborative Filtering (NCF).
The model will learn:
- a trainable embedding for each user;
- a trainable embedding for each game; and
- a multilayer perceptron that learns nonlinear user–game interactions.
The neural model will be evaluated against simpler recommendation baselines.

Check-in 3 — Advanced Extension
The planned advanced extension is a hybrid neural recommender that supplements collaborative signals with Steam game metadata such as:
- genres;
- categories;
- developer/publisher information;
- platform support; and
- potentially textual game descriptions.
This extension will test whether content information improves recommendations, particularly for sparse or less-popular games.

Dataset
Source
The project uses the public Steam User–Game Interactions Dataset hosted on Hugging Face:
https://huggingface.co/datasets/frankjc2022/steam-dataset
Dataset ID:
frankjc2022/steam-dataset
The dataset card lists the dataset under the MIT License.
The source contains four configurations:
- user_games
- games
- reviews
- steamspy_games
Check-in 1 uses only user_games and games.
Downloaded Data
The project download pipeline recorded the following data locally:
Configuration	Rows	Columns	Local Parquet Size
user_games	11,285,290	9	512.48 MB
games	47,811	29	133.21 MB


The large Parquet files are intentionally excluded from Git. The repository contains the code required to reproduce them.
user_games Fields

The interaction table contains:
- steamid
- appid
- game_count
- game_name
- playtime_forever
- img_icon_url
- has_community_visible_stats
- has_leaderboards
- content_descriptorids

For the recommender, the most important variables are:
- steamid — user identifier;
- appid — Steam game identifier;
- game_name — human-readable game name; and
- playtime_forever — lifetime recorded playtime.
games Fields

The metadata table contains 29 fields, including:
- Steam application ID;
- game name and product type;
- free/paid status;
- age requirement;
- short and detailed descriptions;
- supported languages;
- developers and publishers;
- supported platforms;
- categories and genres;
- price information;
- Metacritic information;
- Steam recommendation counts;
- release date; and
- content descriptors.

This metadata makes the same dataset suitable for the planned hybrid-model extension later in the semester.

Dataset Access and Reproduction:
The repository includes:
notebooks/00_download_data.ipynb
This notebook downloads the required Hugging Face configurations and creates:
data/raw/user_games.parquet
data/raw/games.parquet
data/raw/manifest.json
The two Parquet files are excluded from Git because of their size. manifest.json records the downloaded configurations, row counts, column names, and local file sizes.
A minimal Hugging Face example is:
from datasets import load_dataset

DATASET_ID = "frankjc2022/steam-dataset"

user_games = load_dataset(
    DATASET_ID,
    name="user_games",
    split="train",)

games = load_dataset(
    DATASET_ID,
    name="games",
    split="train",)
    
Hugging Face also exposes the source Parquet files directly. For example:
import pandas as pd

df = pd.read_parquet("hf://datasets/frankjc2022/steam-dataset/data/v1/user_games.parquet")

For this project, the preferred workflow is to download the data once with 00_download_data.ipynb and perform subsequent analysis against the local Parquet files.
Exploratory Data Analysis

The Check-in 1 exploratory analysis is contained in:
notebooks/01_eda.ipynb
Because user_games contains more than 11 million interaction records, the notebook uses DuckDB to query the Parquet files directly rather than materializing the full interaction dataset in a Pandas DataFrame.
The EDA includes the following audits.
Interaction Structure
The notebook measures:
- total interaction records;
- unique users;
- unique games represented in the interaction data;
- user × game matrix density and sparsity; and
- metadata coverage between interaction records and the game catalog.
These quantities determine whether collaborative filtering is feasible and quantify 

Playtime:
playtime_forever is evaluated using summary statistics, quantiles, and interpretable playtime buckets:
- 0 minutes;
- 1–60 minutes;
- 1–10 hours;
- 10–100 hours; and
- 100+ hours.

The analysis is important because ownership alone is a weak preference signal. Games with zero recorded playtime may represent purchases that were never played, bundles, gifts, or other interactions that should not automatically be interpreted as positive feedback.
The notebook also compares candidate positive-interaction definitions such as:
- playtime_forever > 0;
- at least 30 minutes;
- at least 1 hour; and
- at least 2 hours.

For the initial modeling stage, playtime_forever > 0 is the planned starting definition because it removes zero-playtime ownership while preserving broad interaction coverage.

User Activity:
The notebook calculates user-level activity including:
- games owned;
- games with positive recorded playtime; and
- total lifetime playtime.
The user-activity distribution is expected to be highly uneven, which creates an important modeling issue: users with very little history provide substantially less information from which to learn an embedding.
This motivates filtering users to a minimum number of positive interactions before model training and evaluation.

Game Popularity and Long-Tail Behavior:
The notebook calculates the number of interacting users per game and examines:
- the most frequently observed games;
- the distribution of game popularity;
- the cumulative share of interactions accounted for by the most-popular games; and
- the long tail of lower-interaction titles.
This analysis is important because a recommender can obtain deceptively strong results by repeatedly recommending globally popular games rather than learning personalized preferences.
For this reason, the final evaluation will include catalog coverage in addition to ranking accuracy.

Missing Values and Duplicate Interactions:
The data audit checks:
- missing values in the interaction table;
- duplicate (steamid, appid) pairs; and
- missing values in important game metadata fields.

Duplicate user–game records will be resolved before train/validation/test creation so that the same user–game pair cannot leak across multiple splits.

Metadata Coverage:
The notebook joins interaction-game IDs to the games table and evaluates whether games appearing in player histories have corresponding metadata.
It also audits fields such as:
- genres;
- developers;
- publishers;
- prices;
- release dates; and
- Metacritic information.
This establishes whether the dataset has sufficient content information for the planned Check-in 3 hybrid model.

Data Artifacts, Biases, and Likely Failure Modes:
The initial data audit identifies several issues that must be considered when interpreting model results.
1. Implicit Feedback
The dataset does not contain an explicit rating for every user–game interaction. Ownership and playtime are behavioral signals rather than direct statements of preference.
2. Zero-Playtime Ownership
A Steam library can include games that a player has never meaningfully played. Treating every owned game as an equally strong positive signal would introduce noise.
3. Popularity Bias
Highly popular games naturally appear in many more user histories. A model may therefore over-recommend blockbuster titles instead of producing genuinely personalized recommendations.
4. Long-Tail Sparsity
Less-popular games contain fewer user interactions and are harder for collaborative models to learn.
5. Low-History Users
Users with very few positive interactions provide limited evidence about their preferences.
6. Cold Start
A pure ID-based collaborative model cannot directly learn useful embeddings for completely unseen users or games.
The planned hybrid model may partially address the new-game side of this problem by incorporating game metadata.
7. No Interaction Timestamp in user_games
The user_games configuration does not provide an interaction timestamp. Therefore, this project cannot honestly use a chronological train/validation/test split from this table alone.
A reproducible per-user holdout strategy will be used instead.
8. Heavy-Tailed Playtime
Lifetime playtime is not an ideal raw regression target because a small number of highly played games can have extremely large values relative to the rest of the data.
The project therefore treats recommendation primarily as an implicit-feedback ranking problem.
9. Metadata Missingness
Some game metadata fields are incomplete. The hybrid model will need explicit handling for missing content features.
10. Potential Popularity Leakage
Metadata fields such as recommendations_total and external review scores may encode popularity or downstream outcomes rather than neutral game content. These features will not automatically be included in the hybrid model without assessing whether they would introduce undesirable leakage or popularity amplification.
Evaluation Plan
Positive Interactions
The initial positive interaction definition is:
playtime_forever > 0

Eligible Users:
For the first modeling experiment, users will be required to have at least 5 positive interactions.
This provides enough history to create training data while reserving positive examples for validation and testing.
Train / Validation / Test Split
Because the dataset does not provide timestamps in user_games, the project will use a reproducible per-user holdout.

For each eligible user:
1 positive interaction -> validation
1 positive interaction -> test
remaining positives     -> training
A fixed random seed will be used:
random_seed = 42
The same split will be used for all baseline and neural models so results are directly comparable.
Negative Sampling
Unobserved user–game pairs cannot safely be interpreted as confirmed dislikes. They may simply represent games the user has never encountered.
For neural training, a subset of unobserved pairs will therefore be sampled as negatives rather than treating the entire missing user–item matrix as explicit negative feedback.
The sampling procedure will be documented and kept consistent when comparing model variants.
Metrics
Recall@10
Measures the proportion of held-out relevant games recovered among the top 10 recommendations.
NDCG@10
Measures recommendation quality while assigning greater value when relevant games appear closer to the top of the ranked list.
This will be a primary metric because recommendation position matters.
Hit Rate@10
Measures whether at least one held-out relevant game appears among a user's top 10 recommendations.
Catalog Coverage
Measures how much of the eligible game catalog receives recommendations across users.
Coverage helps reveal whether a model is obtaining good ranking metrics primarily by recommending the same popular games to everyone.
Planned Baselines
The project will use increasingly expressive models so that the neural model has meaningful points of comparison.
Baseline 1 — Popularity
Recommend the most frequently interacted-with games that the user has not already seen.
This provides a simple but important benchmark and measures how much performance can be obtained without personalization.
Baseline 2 — Collaborative Filtering / Matrix Factorization
Use learned latent user and item factors with a conventional collaborative-filtering objective.
This provides a more meaningful personalized baseline for determining whether the neural interaction function actually adds value.
Planned Neural Approach — Check-in 2
Check-in 2 will implement Neural Collaborative Filtering.
At a high level:
Steam user ID
     |
     v
User embedding
     \
      \
       +--> combined representation --> MLP --> preference score
      /
     /
Game embedding
     ^
     |
Steam app ID
The model will learn trainable low-dimensional representations of users and games. These representations will be combined and passed through a multilayer perceptron to learn nonlinear user–game interactions.
A starting architecture may use:
User embedding
Game embedding
      |
      v
Concatenation
      |
      v
Dense layer + ReLU
      |
      v
Dense layer + ReLU
      |
      v
Interaction score
Architecture size, embedding dimension, regularization, learning rate, and negative-sampling choices will be treated as experimental decisions rather than assumed improvements.
The model will be evaluated on exactly the same held-out interactions as the non-neural baselines.
Planned Advanced Extension — Check-in 3
The planned advanced extension is a hybrid neural recommendation system.
The collaborative model uses only IDs:
user ID + game ID -> learned preference
The hybrid model will additionally represent what the game is:
game ID
genres
categories
platforms
developer / publisher
game description
        |
        v
game representation
This creates a natural Check-in 3 research question:
Does adding game-content information improve top-N recommendation performance, particularly for sparse or less-popular games?

This keeps the advanced extension directly connected to the original recommendation problem rather than changing to a separate task.
Repository Structure
dsan6600-finalproject/
|
├── README.md
├── requirements.txt
├── .gitignore
|
├── notebooks/
│   ├── 00_download_data.ipynb
│   └── 01_eda.ipynb
|
├── src/
│   └── .gitkeep
|
└── data/
    ├── README.md
    └── raw/
        ├── manifest.json
        ├── user_games.parquet    # local only / gitignored
        └── games.parquet         # local only / gitignored
Future check-ins will add preprocessing, baseline, neural-model, and evaluation code as the project develops.

Environment:
The project is developed with Python 3.12 using a Conda/Jupyter environment.