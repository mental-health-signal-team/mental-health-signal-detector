"""Google Drive model loader — downloads models from a publicly shared Drive folder.

Behavior:
    1. If all required artifacts are already present in models_dir → no-op.
    2. If any artifact is missing and folder_id is set → download the entire Drive
       folder to a temporary directory, then copy only the missing artifacts into
       models_dir (existing files are never touched or overwritten).
    3. If artifacts are missing and folder_id is not set → log a warning and return.

Drive folder structure must mirror models/:
    <folder>/
        lr_model.pkl
        tfidf_vectorizer.pkl
        xgb_depression_classifier.pkl
        xgb_tfidf_vectorizer.pkl
        distilbert_hf/        (subfolder)
        mental_roberta_hf/    (subfolder)

The Drive folder must be shared publicly ("Anyone with the link can view").
No authentication is required. folder_id accepts either a bare ID or the full URL.
"""

import logging
import re
import shutil
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Relative paths used to verify that each artifact is fully present.
# For HF directories, model.safetensors is used as a proxy for a complete download.
_REQUIRED_ARTIFACTS = [
    "lr_model.pkl",
    "tfidf_vectorizer.pkl",
    "xgb_depression_classifier.pkl",
    "xgb_tfidf_vectorizer.pkl",
    "distilbert_hf/model.safetensors",
    "mental_roberta_hf/model.safetensors",
]


def ensure_models(models_dir: Path, folder_id: str) -> None:
    """Download missing model artifacts from Google Drive into models_dir.

    Args:
        models_dir: Local models/ directory (created if it does not exist).
        folder_id:  Google Drive folder ID or full folder URL.
                    Pass an empty string to disable Drive download entirely.
    """
    if _all_present(models_dir):
        logger.debug("All models present at %s — skipping download.", models_dir)
        return

    if not folder_id:
        logger.warning(
            "Some models are missing from %s and GDRIVE_MODEL_FOLDER_ID is not set. Model loading will fail if those artifacts are accessed.",
            models_dir,
        )
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    _download_missing(folder_id, models_dir)


def _all_present(models_dir: Path) -> bool:
    return all((models_dir / p).exists() for p in _REQUIRED_ARTIFACTS)


def _find_src_dir(tmp_path: Path) -> Path:
    """Find the downloaded folder root by locating a known artifact inside tmp_path."""
    for match in tmp_path.rglob("lr_model.pkl"):
        return match.parent
    # Fallback: first subdirectory gdown created, or tmp_path itself
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    return subdirs[0] if subdirs else tmp_path


def _extract_id(folder_id_or_url: str) -> str:
    """Extract the bare Drive folder ID from a URL, or return as-is if already an ID."""
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_id_or_url)
    return match.group(1) if match else folder_id_or_url


_MAX_RETRIES = 3
_RETRY_DELAY = 30  # seconds to wait between retries (Drive rate-limit window)


def _download_missing(folder_id: str, models_dir: Path) -> None:
    """Download the Drive folder to a temp dir, then move only missing artifacts."""
    import gdown  # noqa: PLC0415
    from gdown.exceptions import FileURLRetrievalError  # noqa: PLC0415

    bare_id = _extract_id(folder_id)
    logger.info("Downloading models from Google Drive (folder %s)", bare_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                gdown.download_folder(
                    id=bare_id,
                    output=tmp_dir,
                    quiet=False,
                )
                break
            except FileURLRetrievalError:
                if attempt == _MAX_RETRIES:
                    raise
                logger.warning(
                    "Drive rate-limit hit (attempt %d/%d). Retrying in %ds…",
                    attempt,
                    _MAX_RETRIES,
                    _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)

        # gdown creates a subfolder named after the Drive folder inside tmp_dir.
        # Locate the actual root by finding where lr_model.pkl landed.
        src_dir = _find_src_dir(Path(tmp_dir))

        for item in src_dir.iterdir():
            dest = models_dir / item.name
            if dest.exists():
                logger.debug("Skipping existing artifact: %s", item.name)
                continue
            logger.info("Installing artifact: %s", item.name)
            shutil.move(str(item), str(dest))
