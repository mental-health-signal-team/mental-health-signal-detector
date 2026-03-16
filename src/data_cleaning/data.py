import json
import os
from pathlib import Path
from typing import Optional

import kagglehub
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from skimage import data_dir

def download_data() -> None:
    """Load the dataset from the Kaggle API."""
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    data_dir = os.path.abspath("../../data")
    expected_file = os.path.join(data_dir, "reddit_depression_dataset.csv")

    if os.path.exists(expected_file):
        path = data_dir
        print("Dataset déjà téléchargé, téléchargement ignoré.")
    else:
        path = kagglehub.dataset_download("rishabhkausish/reddit-depression-dataset", output_dir=data_dir)
        print("Dataset téléchargé.")

def load_data() -> pd.DataFrame:
    df = pd.read_csv("../../data/reddit_depression_dataset.csv")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicates
    df = df.drop_duplicates()
    #Remove na_values for those who haven't not a lot of na
    df = df.dropna(subset=["Unnamed: 0","subreddit","title",
                       "upvotes", "created_utc", "label"], axis=0)
    #Inversion de certaines valeurs de Unnamed: 0 et title
    unnamed_is_numeric = pd.to_numeric(df["Unnamed: 0"], errors="coerce").notna()
    title_is_numeric = pd.to_numeric(df["title"], errors="coerce").notna()
    mask = (~unnamed_is_numeric) & title_is_numeric

    # 3) Inversion des valeurs entre les 2 colonnes sur ces lignes
    df.loc[mask, ["Unnamed: 0", "title"]] = (
        df.loc[mask, ["title", "Unnamed: 0"]].to_numpy())
    #On drop les colonnes qui ne sont pas utiles
    #On va fusionner les colonnes de texte body et title 
    sub_df = df[["title", "body"]]
    text = sub_df["title"] + " " + sub_df["body"]
    df["title"] = text
    df.drop(columns=["Unnamed: 0", "body", "subreddit", "upvotes",
                     "created_utc", "num_comments"], inplace=True)
    return df

def balance_classes(df, label_col, random_state=42):
    class_counts = df[label_col].value_counts()
    min_count = class_counts.min()
    balanced_df = pd.DataFrame()
    
    for label in class_counts.index:
        label_df = df[df[label_col] == label]
        balanced_label_df = label_df.sample(min_count, random_state=42)
        balanced_df = pd.concat([balanced_df, balanced_label_df])
    
    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)



