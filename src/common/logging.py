"""Structured logging configuration for the Mental Health Signal Detector."""

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """Configure root logger with a structured format.

    Call once at application startup (e.g. in FastAPI lifespan or training scripts).
    Level is read from the LOG_LEVEL environment variable, defaulting to INFO.
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Prefer this over logging.getLogger() for consistency."""
    return logging.getLogger(name)
