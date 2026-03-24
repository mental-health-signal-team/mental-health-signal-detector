"""
One-time conversion: reload mental_roberta_base.pkl with the compatibility
patches and re-save in HuggingFace native format (models/mental_roberta_hf/).

Run once from the project root:
    python scripts/convert_mental_roberta.py
"""

import io
import pickle
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_CLASS_RENAMES = {
    ("transformers.models.roberta.modeling_roberta", "RobertaSdpaSelfAttention"): (
        "transformers.models.roberta.modeling_roberta",
        "RobertaSelfAttention",
    ),
}


class CPUUnpickler(pickle.Unpickler):
    """Custom unpickler to load PyTorch models on CPU, with a patch to handle renamed classes in newer transformers versions."""

    def find_class(self, module, name):
        """Override the find_class method to handle loading PyTorch models on CPU and to account for renamed classes in transformers."""
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        module, name = _CLASS_RENAMES.get((module, name), (module, name))
        return super().find_class(module, name)


def main():
    """Load the old mental_roberta_base.pkl model, extract its state_dict,
    and save it in HuggingFace format for easier future loading and compatibility with newer transformers versions."""
    pkl_path = PROJECT_ROOT / "models" / "mental_roberta_base.pkl"
    output_dir = PROJECT_ROOT / "models" / "mental_roberta_hf"

    print(f"Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        old_model = CPUUnpickler(f).load()

    # state_dict() only reads weights (nn.Parameter / buffers) — safe even if
    # forward() is broken due to the version mismatch.
    state_dict = old_model.state_dict()
    print(f"  {len(state_dict)} weight tensors extracted")
    del old_model

    print("Creating fresh roberta-base skeleton (num_labels=2) ...")
    fresh = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    missing, unexpected = fresh.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING — missing keys: {missing}")
    if unexpected:
        print(f"  WARNING — unexpected keys: {unexpected}")

    print(f"Saving model to {output_dir} ...")
    fresh.save_pretrained(output_dir)

    print("Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokenizer.save_pretrained(output_dir)

    print(f"\nDone. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
