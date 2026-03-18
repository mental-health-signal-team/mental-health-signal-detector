from pathlib import Path

import kagglehub
import pandas as pd

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
    df = pd.read_csv(_get_project_data_dir() / DATA_FILENAME)
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
    df.loc[mask, ["Unnamed: 0", "title"]] = df.loc[
        mask, ["title", "Unnamed: 0"]
    ].to_numpy()
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
    # df = balance_classes(df, label_col="label")
    # #we can do it later if we want to test the model on an unbalanced dataset,
    # which is more realistic

    return df


def balance_classes(df, label_col, random_state=42):
    """Balance the classes in the dataset."""
    class_counts = df[label_col].value_counts()
    min_count = class_counts.min()
    balanced_df = pd.DataFrame()

    for label in class_counts.index:
        label_df = df[df[label_col] == label]
        balanced_label_df = label_df.sample(min_count, random_state=random_state)
        balanced_df = pd.concat([balanced_df, balanced_label_df])

    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
