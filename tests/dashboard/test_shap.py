"""Tests unitaires pour les fonctions SHAP du dashboard."""

import sys
from unittest.mock import MagicMock, Mock

import numpy as np


def test_shap_values():
    """Vérifie que get_shap_values retourne bien un shap.Explainer."""
    mock_shap = MagicMock()
    mock_shap_values = Mock()
    explainer_instance = Mock(return_value=mock_shap_values)
    mock_shap.Explainer.return_value = explainer_instance

    original_shap = sys.modules.get("shap")
    sys.modules["shap"] = mock_shap
    try:
        from src.dashboard.shap import get_shap_values

        model = Mock()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])

        result = get_shap_values(model=model, X_train=X_train, sample_size=None)

        assert result is mock_shap_values
        mock_shap.Explainer.assert_called_once()
        explainer_instance.assert_called_once()
    finally:
        if original_shap is None:
            del sys.modules["shap"]
        else:
            sys.modules["shap"] = original_shap


def test_shap_graph():
    """Vérifie que shap_graph génère bien un graphique."""
    mock_shap = MagicMock()
    mock_shap_values = MagicMock()
    mock_shap_values.__getitem__.return_value = Mock()
    explainer_instance = Mock(return_value=mock_shap_values)
    mock_shap.Explainer.return_value = explainer_instance

    mock_plt = MagicMock()
    mock_matplotlib = MagicMock()
    mock_matplotlib.pyplot = mock_plt

    original_shap = sys.modules.get("shap")
    original_matplotlib = sys.modules.get("matplotlib")
    original_pyplot = sys.modules.get("matplotlib.pyplot")
    sys.modules["shap"] = mock_shap
    sys.modules["matplotlib"] = mock_matplotlib
    sys.modules["matplotlib.pyplot"] = mock_plt
    try:
        from src.dashboard.shap import shap_graph

        model = Mock()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])

        shap_graph(model=model, X_train=X_train, sample_size=None, show=False)

        mock_shap.plots.bar.assert_called_once()
        mock_plt.tight_layout.assert_called_once()
    finally:
        if original_shap is None:
            del sys.modules["shap"]
        else:
            sys.modules["shap"] = original_shap
        if original_matplotlib is None:
            del sys.modules["matplotlib"]
        else:
            sys.modules["matplotlib"] = original_matplotlib
        if original_pyplot is None:
            del sys.modules["matplotlib.pyplot"]
        else:
            sys.modules["matplotlib.pyplot"] = original_pyplot
