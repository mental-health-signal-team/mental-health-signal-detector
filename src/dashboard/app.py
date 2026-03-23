import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

try:
    from src.dashboard.pages import render_prediction_page, render_word_importance_page
except ModuleNotFoundError:
    # Streamlit can run with a cwd that does not include project root in sys.path.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dashboard.pages import render_prediction_page, render_word_importance_page

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL_LOCAL = os.getenv("API_URL_LOCAL", "http://localhost:8000")

def main() -> None:
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Go to",
        ["Prediction", "Word Importance"],
        index=0,
    )

    if selected_page == "Prediction":
        render_prediction_page(API_URL_LOCAL)
    else:
        render_word_importance_page(API_URL_LOCAL)


if __name__ == "__main__":
    main()
