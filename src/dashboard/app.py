import os
import sys
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from src.dashboard.pages import render_prediction_page, render_word_importance_page
except ModuleNotFoundError:
    # Streamlit can launch from a cwd that does not include project root in sys.path.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dashboard.pages import render_prediction_page, render_word_importance_page

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL_LOCAL = os.getenv("API_URL_LOCAL", "http://127.0.0.1:8000")


def _is_api_reachable(api_url: str) -> bool:
    """Check whether the API health endpoint is reachable."""
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        return response.ok
    except requests.exceptions.RequestException:
        return False


def _resolve_api_url() -> str:
    """Use the first reachable endpoint from local and remote candidates."""
    candidates = [
        API_URL_LOCAL,
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        API_URL,
    ]
    unique_candidates = list(dict.fromkeys(url.strip() for url in candidates if url and url.strip()))

    for url in unique_candidates:
        if _is_api_reachable(url):
            return url

    return API_URL_LOCAL


def main() -> None:
    """
    Main entry point for Streamlit dashboard app.
    Resolves API URL, sets up sidebar navigation, and renders selected page."""
    api_url = _resolve_api_url()

    st.sidebar.title("Navigation")
    st.sidebar.caption(f"API endpoint: {api_url}")
    selected_page = st.sidebar.radio(
        "Go to",
        ["Prediction", "Word Importance"],
        index=0,
    )

    if selected_page == "Prediction":
        render_prediction_page(api_url)
    else:
        render_word_importance_page(api_url)


if __name__ == "__main__":
    main()
