"""Train and save only the XGBoost artifacts."""

import src.common.config as config
from src.training.train import load_and_prepare_data, save_artifacts, train_xgboost_model


def main() -> None:
    """Run the XGBoost-only training pipeline and persist artifacts."""
    X_train, y_train, *_ = load_and_prepare_data()
    vectorizer, model = train_xgboost_model(X_train, y_train)
    save_artifacts(
        vectorizer,
        model,
        vectorizer_path=config.XGBOOST_VECTORIZER_PATH,
        model_path=config.XGBOOST_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
