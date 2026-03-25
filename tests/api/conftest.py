import pytest

import src.common.config as config


def pytest_collection_modifyitems(items):
    """Skip all API integration tests when model artifacts are not present."""
    if config.LR_MODEL_PATH.exists():
        return
    skip = pytest.mark.skip(reason="model artifacts not available — skipping API integration tests")
    for item in items:
        if "tests/api" in str(item.fspath):
            item.add_marker(skip)
