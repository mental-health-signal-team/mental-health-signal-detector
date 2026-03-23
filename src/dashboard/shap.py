from typing import Any


def get_shap_values(
    model: Any,
    X_train: Any,
    sample_size: int = 10000,
    random_state: int = 42,
):
    """Compute SHAP values from an existing model and training matrix.
    WARNING : CRASH IF THE TRAINING MATRIX IS TOO LARGE. USE A SAMPLE OF THE TRAINING MATRIX IF NECESSARY."""
    import numpy as np

    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Install it with `pip install shap` to use this function.") from exc

    if model is None:
        raise ValueError("model cannot be None.")
    if X_train is None:
        raise ValueError("X_train cannot be None.")

    n_rows = X_train.shape[0]
    if n_rows == 0:
        raise ValueError("X_train is empty; cannot compute SHAP values.")

    if sample_size is None:
        X_sample = X_train
    else:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer when provided.")
        if sample_size > n_rows:
            raise ValueError(f"sample_size ({sample_size}) cannot exceed number of rows in X_train ({n_rows}).")

        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n_rows, size=sample_size, replace=False)
        if hasattr(X_train, "iloc"):
            X_sample = X_train.iloc[sample_idx]
        else:
            X_sample = X_train[sample_idx]

    explainer = shap.Explainer(model, X_sample)
    return explainer(X_sample)


def shap_graph(
    model: Any,
    X_train: Any,
    vectorizer: Any | None = None,
    sample_size: int = 10000,
    max_display: int = 20,
    show: bool = True,
    random_state: int = 42,
):
    """Plot a SHAP bar graph from precomputed SHAP values.
    WARNING : CRASH IF THE TRAINING MATRIX IS TOO LARGE. USE A SAMPLE OF THE TRAINING MATRIX IF NECESSARY."""
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Install it with `pip install shap` to use this function.") from exc

    n_rows = X_train.shape[0]
    if n_rows == 0:
        raise ValueError("X_train is empty; cannot compute SHAP values.")

    if sample_size is None:
        X_sample = X_train
    else:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer when provided.")
        if sample_size > n_rows:
            raise ValueError(f"sample_size ({sample_size}) cannot exceed number of rows in X_train ({n_rows}).")

        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n_rows, size=sample_size, replace=False)
        if hasattr(X_train, "iloc"):
            X_sample = X_train.iloc[sample_idx]
        else:
            X_sample = X_train[sample_idx]

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    if vectorizer is not None and hasattr(vectorizer, "get_feature_names_out"):
        shap_values.feature_names = vectorizer.get_feature_names_out().tolist()

    shap.plots.bar(shap_values[0], max_display=max_display)
    plt.tight_layout()
    if show:
        plt.show()
