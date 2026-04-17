"""Smoke tests for the Streamlit dashboard application module."""

import importlib


def test_dashboard_app_imports() -> None:
    """The dashboard app module must be importable without errors."""
    mod = importlib.import_module("src.dashboard.app")
    assert mod is not None


def test_dashboard_pages_imports() -> None:
    """The dashboard pages module must be importable without errors."""
    mod = importlib.import_module("src.dashboard.pages")
    assert mod is not None


def test_dashboard_stats_imports() -> None:
    """The dashboard stats module must be importable without errors."""
    mod = importlib.import_module("src.dashboard.stats")
    assert mod is not None
