"""
Pipeline de chargement et nettoyage des datasets.

Datasets supportés :
  - Kaggle reddit depression dataset (CSV local — 2.47M posts Reddit)
  - DAIR-AI/emotion (HuggingFace — 18K phrases labelisées)
  - GoEmotions (Google/HuggingFace — 54K commentaires Reddit, 28 émotions)
  - SMHD (si disponible — 9 troubles mentaux, accès sur demande)
"""

import re
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from loguru import logger


# ---------------------------------------------------------------------------
# Nettoyage texte
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Nettoie un texte brut : supprime URLs, mentions, caractères spéciaux."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"'", "", text)                        # contractions : I'm→Im, don't→dont
    text = re.sub(r"[^a-zA-Z\s]", " ", text)            # caractères non-alpha
    text = re.sub(r"\s+", " ", text).strip()             # espaces multiples
    return text.lower()


# ---------------------------------------------------------------------------
# Chargement datasets
# ---------------------------------------------------------------------------

def load_kaggle_depression(
    csv_path: str | Path,
    max_samples: int | None = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Charge le dataset Kaggle reddit depression (2.47M posts).

    Colonnes du CSV : subreddit, title, body, upvotes, created_utc, num_comments, label
    - Combine title + body pour avoir le texte complet
    - Sous-échantillonne pour équilibrer les classes (80/20 → ~60/40)
    - max_samples : nb total de lignes à garder (None = tout)
    """
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Kaggle brut : {len(df)} lignes | colonnes={list(df.columns)}")

    # Combine title + body
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = (df["title"] + " " + df["body"]).str.strip()

    df = df[["text", "label"]].dropna(subset=["label"])
    df["label"] = df["label"].astype(float).astype(int)
    df = df[df["text"].str.len() > 20]

    # Rééquilibrage : on garde tous les posts détresse (label=1)
    # et on sous-échantillonne les non-détresse pour ratio ~60/40
    df_distress = df[df["label"] == 1]
    df_normal = df[df["label"] == 0]

    n_distress = len(df_distress)
    n_normal_target = int(n_distress * 1.5)  # ratio 40% détresse / 60% non-détresse
    df_normal = df_normal.sample(n=min(n_normal_target, len(df_normal)), random_state=random_state)

    df = pd.concat([df_distress, df_normal], ignore_index=True)

    # Sous-échantillonnage global si demandé (stratifié)
    if max_samples and len(df) > max_samples:
        from sklearn.model_selection import train_test_split as _split
        df, _ = _split(df, train_size=max_samples, random_state=random_state, stratify=df["label"])

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10]

    logger.info(f"Kaggle après nettoyage : {len(df)} lignes | distribution:\n{df['label'].value_counts()}")
    return df.reset_index(drop=True)


def load_dair_emotion(split: str = "train") -> pd.DataFrame:
    """
    Charge DAIR-AI/emotion depuis HuggingFace.
    Labels originaux (6 classes) → binaire : sadness(0) + fear(2) = détresse=1, reste=0.
    """
    dataset = load_dataset("dair-ai/emotion", split=split)
    df = pd.DataFrame({"text": dataset["text"], "label_orig": dataset["label"]})

    # sadness=0, joy=1, love=2, anger=3, fear=4, surprise=5
    DISTRESS_LABELS = {0, 4}  # sadness + fear
    df["label"] = df["label_orig"].apply(lambda x: 1 if x in DISTRESS_LABELS else 0)
    df["text"] = df["text"].apply(clean_text)
    df = df[["text", "label"]]

    logger.info(f"DAIR-AI/emotion [{split}] : {len(df)} lignes | distribution:\n{df['label'].value_counts()}")
    return df.reset_index(drop=True)


def load_go_emotions(split: str = "train") -> pd.DataFrame:
    """
    Charge GoEmotions (Google) depuis HuggingFace.
    54K commentaires Reddit avec 28 labels d'émotions.

    Binarisation cliniquement motivée :
      Détresse (1) = sadness, grief, fear, nervousness, remorse, embarrassment, disappointment
      Non-détresse (0) = tout le reste (joy, admiration, neutral, etc.)
    """
    # Labels de détresse clinique (index GoEmotions simplified)
    # 9=disappointment, 12=embarrassment, 14=fear, 16=grief, 19=nervousness, 24=remorse, 25=sadness
    DISTRESS_LABELS = {9, 12, 14, 16, 19, 24, 25}

    dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    df = pd.DataFrame({"text": dataset["text"], "labels": dataset["labels"]})

    df["label"] = df["labels"].apply(lambda x: 1 if any(lbl in DISTRESS_LABELS for lbl in x) else 0)
    df["text"] = df["text"].apply(clean_text)
    df = df[["text", "label"]]
    df = df[df["text"].str.len() > 10]

    logger.info(f"GoEmotions [{split}] : {len(df)} lignes | distribution:\n{df['label'].value_counts()}")
    return df.reset_index(drop=True)


def load_smhd(csv_path: str | Path) -> pd.DataFrame:
    """
    Charge le dataset SMHD (Self-reported Mental Health Diagnoses).
    Format attendu : colonnes 'post' (texte) et 'label' (0/1).
    À activer quand l'accès est obtenu.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"SMHD brut : {len(df)} lignes | colonnes={list(df.columns)}")

    col_map = {}
    for col in df.columns:
        if col.lower() in ("post", "text", "body", "content"):
            col_map[col] = "text"
        elif col.lower() in ("label", "diagnosis", "target"):
            col_map[col] = "label"
    df = df.rename(columns=col_map)[["text", "label"]]
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 10]

    logger.info(f"SMHD après nettoyage : {len(df)} lignes | distribution:\n{df['label'].value_counts()}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge + split
# ---------------------------------------------------------------------------

def build_dataset(
    kaggle_path: str | Path | None = None,
    use_dair: bool = True,
    use_go_emotions: bool = False,
    smhd_path: str | Path | None = None,
    kaggle_max_samples: int = 100_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble les sources disponibles et retourne (train_df, test_df).

    Priorité des sources :
      1. Kaggle reddit depression (vrai Reddit, 2.47M posts)
      2. DAIR-AI/emotion (diversité stylistique)
      3. GoEmotions (Reddit, labels fins, qualité Google)
      4. SMHD (si accès obtenu — qualité clinique)
    """
    from sklearn.model_selection import train_test_split

    frames = []
    if kaggle_path:
        frames.append(load_kaggle_depression(kaggle_path, max_samples=kaggle_max_samples))
    if use_dair:
        frames.append(load_dair_emotion("train"))
        frames.append(load_dair_emotion("test"))
    if use_go_emotions:
        frames.append(load_go_emotions("train"))
        frames.append(load_go_emotions("validation"))
        frames.append(load_go_emotions("test"))
    if smhd_path:
        frames.append(load_smhd(smhd_path))

    if not frames:
        raise ValueError("Aucune source de données fournie.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    logger.info(f"Dataset combiné : {len(df)} lignes | distribution:\n{df['label'].value_counts()}")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
