import os
from datasets import load_dataset

def download_emotion_dataset(target_dir="data/raw"):
    os.makedirs(target_dir, exist_ok=True)
    print("Downloading dair-ai/emotion dataset from Hugging Face...")
    dataset = load_dataset("emotion")
    for split in dataset:
        out_path = os.path.join(target_dir, f"emotion_{split}.csv")
        dataset[split].to_csv(out_path, index=False)
        print(f"Saved {split} split to {out_path}")

if __name__ == "__main__":
    download_emotion_dataset()
