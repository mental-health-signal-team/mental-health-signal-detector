import os
import sys
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from src.dashboard.about import render_about_page
    from src.dashboard.pages import render_models_board_page, render_prediction_page, render_word_importance_page
    from src.dashboard.stats import render_stats_page
except ModuleNotFoundError:
    # Streamlit can launch from a cwd that does not include project root in sys.path.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dashboard.about import render_about_page
    from src.dashboard.pages import render_models_board_page, render_prediction_page, render_word_importance_page
    from src.dashboard.stats import render_stats_page

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
                overflow: hidden;
                background:
                    radial-gradient(circle at 88% 10%, rgba(14, 199, 230, 0.18) 0 12rem, transparent 12rem),
                    radial-gradient(circle at 95% 78%, rgba(14, 199, 230, 0.22) 0 9rem, transparent 9rem),
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
                background: transparent;
                color: var(--cyan-500);
                letter-spacing: 0.1em;
                text-transform: uppercase;
                font-weight: 700;
                font-size: 0.72rem;
                padding: 0.4rem 0;
                border-bottom: 1px solid rgba(66, 216, 240, 0.25);
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
                background: rgba(14, 199, 230, 0.08);
                color: var(--cyan-400);
                font-weight: 700;
                padding: 0.38rem 0.8rem;
                border-radius: 0.25rem;
                border: 1px solid rgba(66, 216, 240, 0.35);
                font-size: 0.82rem;
                letter-spacing: 0.02em;
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

            /* ── Sidebar shell ── */
            section[data-testid="stSidebar"],
            div[data-testid="stSidebar"],
            div[data-testid="stSidebar"] > div,
            div[data-testid="stSidebar"] > div > div {
                background: linear-gradient(180deg, #020e1e 0%, #041527 60%, #051d38 100%) !important;
                border-right: 1px solid rgba(66, 216, 240, 0.18) !important;
            }
            div[data-testid="stSidebar"] h1,
            div[data-testid="stSidebar"] p,
            div[data-testid="stSidebar"] label {
                color: var(--ink-100);
            }

            /* ── Sidebar brand block ── */
            .sb-brand {
                padding: 1.4rem 1.1rem 1.1rem;
                border-bottom: 1px solid rgba(66, 216, 240, 0.15);
                margin-bottom: 0.5rem;
            }
            .sb-brand-eyebrow {
                font-size: 0.62rem;
                font-weight: 800;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: var(--cyan-500);
                margin-bottom: 0.45rem;
            }
            .sb-brand-title {
                font-size: 1.15rem;
                font-weight: 900;
                line-height: 1.15;
                color: #ffffff;
                letter-spacing: 0.01em;
            }
            .sb-brand-sub {
                font-size: 0.73rem;
                color: var(--ink-200);
                margin-top: 0.35rem;
                font-weight: 500;
            }

            /* ── Nav section label ── */
            .sb-nav-label {
                font-size: 0.62rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: rgba(66, 216, 240, 0.55);
                padding: 0.6rem 1.1rem 0.3rem;
            }

            /* ── Sidebar nav buttons styled as links ── */
            div[data-testid="stSidebar"] .stButton > button {
                background: transparent !important;
                color: var(--ink-200) !important;
                border: none !important;
                border-left: 3px solid transparent !important;
                border-radius: 0.4rem !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
                text-align: left !important;
                justify-content: flex-start !important;
                padding: 0.6rem 0.85rem !important;
                box-shadow: none !important;
                text-transform: none !important;
                letter-spacing: normal !important;
            }
            div[data-testid="stSidebar"] .stButton > button:hover {
                background: rgba(14, 199, 230, 0.08) !important;
                border-left-color: rgba(66, 216, 240, 0.45) !important;
                color: var(--ink-100) !important;
                transform: none !important;
                filter: none !important;
            }
            /* Active nav item uses type="primary" — higher specificity than the general rule */
            div[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
                background: rgba(14, 199, 230, 0.13) !important;
                border-left-color: var(--cyan-500) !important;
                color: #ffffff !important;
            }
            div[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]:hover {
                background: rgba(14, 199, 230, 0.18) !important;
                transform: none !important;
                filter: none !important;
            }

            /* ── Sidebar footer card (API status) ── */
            .sb-footer {
                margin: 1.2rem 0.55rem 0.8rem;
                background: rgba(4, 21, 43, 0.7);
                border: 1px solid rgba(66, 216, 240, 0.15);
                border-radius: 0.4rem;
                padding: 0.7rem 0.9rem;
            }
            .sb-footer-label {
                font-size: 0.6rem;
                font-weight: 800;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: rgba(66, 216, 240, 0.5);
                margin-bottom: 0.3rem;
            }
            .sb-footer-dot {
                display: inline-block;
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: #2ecc71;
                margin-right: 0.4rem;
                box-shadow: 0 0 5px #2ecc71;
                vertical-align: middle;
            }
            .sb-footer-status {
                font-size: 0.73rem;
                color: #7be5a0;
                font-weight: 600;
            }
            .sb-footer-url {
                font-size: 0.68rem;
                color: var(--ink-200);
                margin-top: 0.2rem;
                word-break: break-all;
            }

            .stTextArea textarea,
            .stSelectbox [data-baseweb="select"] > div,
            .stSlider [data-baseweb="slider"] {
                background: rgba(8, 28, 52, 0.86);
                border: 1px solid rgba(66, 216, 240, 0.35);
                color: var(--ink-100);
            }

            /* ── Widget labels (textarea, selectbox, slider, radio…) ── */
            .stTextArea label,
            .stSelectbox label,
            .stSlider label,
            .stRadio label,
            .stCheckbox label,
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] span {
                color: var(--ink-100) !important;
            }

            /* ── Markdown headings (### used inline in pages) ── */
            .stMarkdown h2,
            .stMarkdown h3,
            .stMarkdown h4 {
                color: var(--ink-100);
            }

            /* ── Captions ── */
            [data-testid="stCaptionContainer"] p,
            .stCaption p {
                color: var(--ink-200) !important;
            }

            /* ── Metric cards (st.metric) ── */
            [data-testid="stMetricLabel"] p {
                color: var(--ink-200) !important;
            }
            [data-testid="stMetricValue"] {
                color: #ffffff !important;
            }
            [data-testid="stMetricDelta"] {
                color: var(--cyan-400) !important;
            }

            /* ── st.table() ── */
            .stTable table {
                background: rgba(4, 21, 43, 0.92);
                border-collapse: collapse;
                width: 100%;
            }
            .stTable th {
                color: var(--cyan-400) !important;
                background: rgba(7, 35, 65, 0.95) !important;
                border-bottom: 1px solid rgba(66, 216, 240, 0.3) !important;
                font-weight: 700;
                text-transform: uppercase;
                font-size: 0.78rem;
                letter-spacing: 0.04em;
                padding: 0.55rem 0.8rem !important;
            }
            .stTable td {
                color: var(--ink-100) !important;
                border-bottom: 1px solid rgba(66, 216, 240, 0.1) !important;
                padding: 0.5rem 0.8rem !important;
            }
            .stTable tr:hover td {
                background: rgba(14, 199, 230, 0.06) !important;
            }

            /* ── Sidebar text (captions, small text) ── */
            div[data-testid="stSidebar"] small {
                color: var(--ink-200) !important;
            }

            /* ── Info / warning / error / success boxes ── */
            [data-testid="stAlert"] p,
            [data-testid="stAlert"] div {
                color: inherit;
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
                    font-size: 0.62rem;
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

    api_url = _resolve_api_url()

    # ── Sidebar brand ──────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div class="sb-brand">
            <div class="sb-brand-eyebrow">Artefact School of Data</div>
            <div class="sb-brand-title">Mental Health<br/>Signal Detector</div>
            <div class="sb-brand-sub">Bootcamp Data Science &mdash; Final Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Navigation ─────────────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="sb-nav-label">Navigate</div>', unsafe_allow_html=True)

    _NAV_ITEMS = [
        ("prediction", "🔮", "Prediction"),
        ("word-importance", "🔍", "Word Importance"),
        ("models-board", "📊", "Models Board"),
        ("stats", "📈", "Stats"),
        ("about", "ℹ️", "About the Models"),
    ]

    if "page" not in st.session_state:
        st.session_state.page = "prediction"
    current_page = st.session_state.page

    for key, icon, label in _NAV_ITEMS:
        btn_type = "primary" if current_page == key else "secondary"
        if st.sidebar.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True, type=btn_type):
            st.session_state.page = key
            st.rerun()

    # ── API status footer ──────────────────────────────────────────────────────
    st.sidebar.markdown(
        f"""
        <div class="sb-footer">
            <div class="sb-footer-label">API Status</div>
            <div class="sb-footer-status">
                <span class="sb-footer-dot"></span>Connected
            </div>
            <div class="sb-footer-url">{api_url}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Team ───────────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="margin:0.8rem 0.55rem 0;padding:0.7rem 0.9rem;
                    background:rgba(4,21,43,0.7);border:1px solid rgba(66,216,240,0.15);
                    border-radius:0.4rem;">
            <div style="font-size:0.6rem;font-weight:800;letter-spacing:0.1em;
                        text-transform:uppercase;color:rgba(66,216,240,0.5);margin-bottom:0.45rem;">
                Team
            </div>
            <div style="font-size:0.78rem;color:#bad7eb;line-height:1.8;">
                Aïmen El Abidi<br/>
                Thomas Dinh<br/>
                Stanislav Grinchenko<br/>
                Fabrice Moncaut
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if current_page == "word-importance":
        render_word_importance_page(api_url)
    elif current_page == "models-board":
        render_models_board_page(api_url)
    elif current_page == "stats":
        render_stats_page(api_url)
    elif current_page == "about":
        render_about_page()
    else:
        render_prediction_page(api_url)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
