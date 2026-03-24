import os
import sys
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from src.dashboard.pages import render_models_board_page, render_prediction_page, render_word_importance_page
except ModuleNotFoundError:
    # Streamlit can launch from a cwd that does not include project root in sys.path.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dashboard.pages import render_models_board_page, render_prediction_page, render_word_importance_page

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL_LOCAL = os.getenv("API_URL_LOCAL", "http://127.0.0.1:8000")


def _inject_theme() -> None:
    """Inject custom CSS inspired by the project visual identity."""
    st.markdown(
        """
        <style>
            :root {
                --navy-900: #04152b;
                --navy-800: #072341;
                --cyan-500: #0ec7e6;
                --cyan-400: #42d8f0;
                --ink-100: #eaf5ff;
                --ink-200: #bad7eb;
            }

            .stApp {
                background:
                    radial-gradient(circle at 88% 10%, rgba(14, 199, 230, 0.25) 0 12rem, transparent 12rem),
                    radial-gradient(circle at 95% 78%, rgba(14, 199, 230, 0.32) 0 9rem, transparent 9rem),
                    linear-gradient(120deg, var(--navy-900) 0%, #031022 45%, var(--navy-800) 100%);
                color: var(--ink-100);
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            }

            .main .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
                max-width: 70rem;
            }

            .app-frame {
                border-left: 10px solid var(--cyan-500);
                padding-left: 1.1rem;
                margin-bottom: 1.2rem;
            }

            .hero-banner {
                background: linear-gradient(90deg, #0aaac9 0%, #16c3df 70%, #2acfea 100%);
                color: #05233e;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                font-weight: 800;
                font-size: 0.75rem;
                padding: 0.55rem 0.9rem;
                border-radius: 0.15rem;
                box-shadow: 0 6px 10px rgba(0, 0, 0, 0.25);
                margin-bottom: 1rem;
            }

            .hero-title {
                margin: 0;
                color: #ffffff;
                font-size: clamp(2rem, 4.5vw, 3.5rem);
                line-height: 1.02;
                letter-spacing: 0.01em;
                font-weight: 900;
            }

            .hero-line {
                display: block;
            }

            .hero-subtitle {
                color: var(--ink-200);
                font-size: 1.1rem;
                margin-top: 0.9rem;
                margin-bottom: 1.1rem;
                max-width: 46rem;
            }

            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.7rem;
                margin: 0.7rem 0 1.6rem;
            }

            .chip {
                background: linear-gradient(90deg, #00a8c7, #10c5e2);
                color: #06324b;
                font-weight: 800;
                padding: 0.42rem 0.8rem;
                border-radius: 0.2rem;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.28);
            }

            .section-title {
                color: var(--cyan-400);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-size: 0.82rem;
                margin-top: 0.3rem;
                margin-bottom: 0.2rem;
                font-weight: 800;
            }

            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #041226 0%, #051d38 100%);
                border-right: 1px solid rgba(66, 216, 240, 0.2);
            }

            div[data-testid="stSidebar"] h1,
            div[data-testid="stSidebar"] p,
            div[data-testid="stSidebar"] label {
                color: var(--ink-100);
            }

            .stTextArea textarea,
            .stSelectbox [data-baseweb="select"] > div,
            .stSlider [data-baseweb="slider"] {
                background: rgba(8, 28, 52, 0.86);
                border: 1px solid rgba(66, 216, 240, 0.35);
                color: var(--ink-100);
            }

            .stButton > button {
                background: linear-gradient(90deg, #09abc9 0%, #24cfea 100%);
                color: #04253e;
                border: none;
                border-radius: 0.22rem;
                font-weight: 800;
                letter-spacing: 0.03em;
                text-transform: uppercase;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.28);
            }

            .stButton > button:hover {
                filter: brightness(1.06);
                transform: translateY(-1px);
            }

            @media (max-width: 900px) {
                .app-frame {
                    border-left-width: 6px;
                    padding-left: 0.75rem;
                }

                .hero-banner {
                    font-size: 0.66rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_shell() -> None:
    """Render reusable shell elements around the page content."""
    st.markdown('<div class="app-frame">', unsafe_allow_html=True)


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

    return API_URL_LOCAL  # On laisse comme ça pour l'instant


def main() -> None:
    """
    Main entry point for Streamlit dashboard app.
    Resolves API URL, sets up sidebar navigation, and renders selected page."""
    st.set_page_config(
        page_title="Mental Health Signal Detector",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()
    _render_shell()

    api_url = API_URL_LOCAL

    st.sidebar.markdown(
        '<h2 style="margin:0; line-height:1.05; color:#eaf5ff;">Mental Health<br/>Signal Detector</h2>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("Bootcamp Data Science")
    st.sidebar.markdown("---")
    st.sidebar.title("Navigation")
    st.sidebar.caption(f"API endpoint: {api_url}")
    selected_page = st.sidebar.radio(
        "Go to",
        ["Prediction", "Word Importance", "Models Board"],
        index=0,
    )

    if selected_page == "Prediction":
        render_prediction_page(api_url)
    elif selected_page == "Word Importance":
        render_word_importance_page(api_url)
    else:
        render_models_board_page(api_url)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
