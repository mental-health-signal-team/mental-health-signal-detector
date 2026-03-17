import os

from datasets import load_dataset


def download_emotion_dataset(target_dir="data/raw"):
    """
    Download the emotion dataset from Hugging Face and save it to a CSV file.
    """
    os.makedirs(target_dir, exist_ok=True)
    print("Downloading dair-ai/emotion unsplit dataset from Hugging Face...")
    dataset = load_dataset("emotion", "unsplit")

    out_path = os.path.join(target_dir, "emotion_unsplit.csv")
    dataset["train"].to_csv(out_path, index=False)
    print(f"Saved unsplit split to {out_path}")
    print(f"Rows saved: {dataset['train'].num_rows}")


if __name__ == "__main__":
    download_emotion_dataset()
