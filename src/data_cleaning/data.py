from pathlib import Path

import kagglehub
import pandas as pd

import src.common.config as config

DATA_FILENAME = "reddit_depression_dataset.csv"


def _get_project_data_dir() -> Path:
    """Resolve the data directory independently from the current working directory."""
    current_file = Path(__file__).resolve()

    # Prefer a location that already contains the target dataset file.
    for parent in current_file.parents:
        candidate = parent / "data"
        if (candidate / DATA_FILENAME).exists():
            return candidate

    # Fallback to the first existing "data" directory found while walking up.
    for parent in current_file.parents:
        candidate = parent / "data"
        if candidate.exists():
            return candidate

    # Last resort: create a data directory at the project level.
    fallback = current_file.parents[2] / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def download_data() -> None:
    """Load the dataset from the Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    data_dir = _get_project_data_dir()
    expected_file = data_dir / DATA_FILENAME

    if expected_file.exists():
        print("Dataset déjà téléchargé, téléchargement ignoré.")
    else:
        kagglehub.dataset_download(
            "rishabhkausish/reddit-depression-dataset",
            output_dir=str(data_dir),
        )
        print("Dataset téléchargé.")


def load_data() -> pd.DataFrame:
    """Load the dataset from the CSV file."""
    df = pd.read_csv(config.DATA_DIR / DATA_FILENAME)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by removing duplicates, handling missing values,
    and balancing classes."""
    # Remove duplicates
    df = df.drop_duplicates()
    # Remove na_values for those who haven't not a lot of na
    df = df.dropna(
        subset=["Unnamed: 0", "subreddit", "title", "upvotes", "created_utc", "label"],
        axis=0,
    )
    # Inversion de certaines valeurs de Unnamed: 0 et title
    unnamed_is_numeric = pd.to_numeric(df["Unnamed: 0"], errors="coerce").notna()
    title_is_numeric = pd.to_numeric(df["title"], errors="coerce").notna()
    mask = (~unnamed_is_numeric) & title_is_numeric

    # 3) Inversion des valeurs entre les 2 colonnes sur ces lignes
    df.loc[mask, ["Unnamed: 0", "title"]] = df.loc[mask, ["title", "Unnamed: 0"]].to_numpy()
    df.loc[mask, ["Unnamed: 0", "title"]] = df.loc[mask, ["title", "Unnamed: 0"]].to_numpy()
    # On drop les colonnes qui ne sont pas utiles
    # On va fusionner les colonnes de texte body et title
    sub_df = df[["title", "body"]]
    text = sub_df["title"].fillna("") + " " + sub_df["body"].fillna("")
    df["title"] = text
    df.drop(
        columns=[
            "Unnamed: 0",
            "body",
            "subreddit",
            "upvotes",
            "created_utc",
            "num_comments",
        ],
        inplace=True,
    )

    df.dropna(inplace=True)
    return df
