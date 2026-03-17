# Mental Health Signal Detector — Project Plan

## Project Overview
Group project (Artefact School of Data) to detect mental health signals (primarily depression) from text data using NLP and machine learning.

---

## Dataset Summary (`data/raw/`)

| File | Rows | Task | Notes |
|---|---|---|---|
| `depression_dataset_reddit_cleaned.csv` | 7,731 | Binary depression (0/1) | Pre-cleaned, balanced ~50/50. Best starting point. |
| `reddit_depression_dataset.csv` | 8.3M | Binary depression (0/1) | Raw Reddit posts, imbalanced (~1:4). Used as primary dataset. |
| `emotion_train/test/validation.csv` | 16K/2K/2K | 6-class emotion | Labels: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise |
| `emotion_unsplit.csv` | 416K | 6-class emotion | Full unsplit version of the above |
| `2018-E-c-En-dev.txt` | 887 | Multi-label emotion (11 flags) | SemEval 2018 Task 1. Useful for eval/augmentation. |

**Recommended strategy:**
- Core model: `reddit_depression_dataset.csv` cleaned + balanced
- Enrichment: emotion datasets for auxiliary features or multi-task learning

---

## Data Pipeline (`src/data_cleaning/data.py`)

### `clean_data(df)` steps:
1. Drop duplicates
2. Drop rows with NaN in key columns (label, title, subreddit, etc.)
3. Fix swapped `Unnamed: 0` / `title` column values
4. Merge `title` + `body` into single `title` column
5. Drop unused columns
6. `dropna()` to remove rows where merged text is NaN
7. `balance_classes()` — undersample majority class to match minority

### Known issues fixed:
- `balance_classes` had hardcoded `random_state=42` instead of using the parameter → fixed
- Module cached in Jupyter after edits → use `importlib.reload()` or restart kernel
- Kaggle credentials must be in `~/.kaggle/kaggle.json` (chmod 600)
- `pyproject.toml` needed `[tool.poetry] package-mode = false` to allow `poetry install`
- Notebooks run from `notebooks/` dir → add `sys.path.insert(0, '..')` to import `src`

---

## Preprocessing Plan for Logistic Regression

Input: `df_cleaned` with columns `title` (text) and `label` (0.0/1.0 float)

### Step 1 — Text Cleaning
- Lowercase
- Remove URLs, HTML tags, Reddit artifacts (`r/`, `u/`)
- Remove punctuation, special characters, numbers (test both with/without)
- Strip extra whitespace

### Step 2 — Tokenization & Stopword Removal
- Tokenize into words
- Remove English stopwords — **keep negations** (not, never, no) — they carry sentiment signal

### Step 3 — Normalization
- Lemmatization (preferred over stemming for interpretability)
- e.g. "feeling" → "feel"

### Step 4 — TF-IDF Vectorization
- `TfidfVectorizer` with:
  - `ngram_range=(1, 2)` — unigrams + bigrams (captures "feel hopeless", "can't cope")
  - `max_features=50_000`
  - `min_df=5` — ignore very rare terms
  - `sublinear_tf=True` — dampen high-frequency term dominance
- Output: sparse matrix `X`

### Step 5 — Label Preparation
- Cast `label` float → int → `y`
- Verify ~50/50 balance

### Step 6 — Train/Val/Test Split
- 70% / 15% / 15%
- `stratify=y` to preserve class balance

### Step 7 — Scaling
- TF-IDF is L2-normalized by default → no additional scaling needed for logistic regression on sparse input

### Output
`(X_train, y_train)`, `(X_val, y_val)`, `(X_test, y_test)` — ready for modeling

---

## Key Reflections

- The `reddit_depression_dataset.csv` subreddits are: `teenagers` (1.95M), `depression` (290K), `SuicideWatch` (190K), `happy` (24K), `DeepThoughts` (9K) — label distribution is heavily skewed toward non-depression (teenagers/happy)
- Text quality varies a lot — Reddit posts can be very long or very short; may need length filtering
- Emotion features (sadness, fear, pessimism from SemEval) could serve as engineered features alongside TF-IDF
- Multi-task approach (predict emotion + depression jointly) is a possible future direction
