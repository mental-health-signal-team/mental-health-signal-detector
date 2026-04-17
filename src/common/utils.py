"""Shared utility functions used across the API, dashboard, and training modules."""


def truncate_text(text: str, max_chars: int = 512) -> str:
    """Truncate text to max_chars, appending '…' if truncated.

    Used to cap inputs before sending to transformer models that have a token limit.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def safe_float(value: object, default: float = 0.0) -> float:
    """Convert value to float, returning default on failure.

    Useful when reading untrusted data (API params, CSV columns) without try/except noise.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))
