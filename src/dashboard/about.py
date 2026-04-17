"""Dashboard About page — renders project description, model benchmark table, and team info."""

import streamlit as st

_MODELS = [
    {
        "name": "Logistic Regression",
        "badge": "Classical ML",
        "badge_color": "#1a6e8e",
        "key": "lr",
        "description": (
            "TF-IDF bag-of-words (50 000 features, unigrams + bigrams) paired with a "
            "regularised logistic regression. Fast, fully interpretable — every prediction "
            "can be traced back to individual tokens via coefficient weights."
        ),
        "training_data": "30 K balanced samples",
        "accuracy": "0.93",
        "precision": "0.93",
        "recall": "0.92",
        "f1": "0.93",
        "speed": "< 5 ms",
        "explainability": "Full (TF-IDF coefficients)",
    },
    {
        "name": "XGBoost",
        "badge": "Classical ML",
        "badge_color": "#1a6e8e",
        "key": "xgboost",
        "description": (
            "Gradient-boosted trees on top of a TF-IDF matrix (10 000 features). "
            "Captures non-linear interactions between tokens that a linear model misses, "
            "with tuned hyperparameters from grid search."
        ),
        "training_data": "30 K balanced samples",
        "accuracy": "0.91",
        "precision": "0.73",
        "recall": "0.87",
        "f1": "0.80",
        "speed": "< 10 ms",
        "explainability": "Partial (feature importance)",
    },
    {
        "name": "DistilBERT",
        "badge": "Transformer",
        "badge_color": "#0e7a5c",
        "key": "distilbert",
        "description": (
            "DistilBERT-base-uncased fine-tuned for binary sequence classification. "
            "Understands word order and context — 'not happy' ≠ 'happy'. "
            "Trained with weighted cross-entropy to handle class imbalance."
        ),
        "training_data": "30 K balanced samples",
        "accuracy": "0.96",
        "precision": "0.94",
        "recall": "0.97",
        "f1": "0.96",
        "speed": "200–400 ms",
        "explainability": "Gradient × input attribution",
    },
    {
        "name": "MentalRoBERTa",
        "badge": "Domain Transformer",
        "badge_color": "#6b3fa0",
        "key": "mental_roberta",
        "description": (
            "RoBERTa pretrained on mental-health corpora, then fine-tuned on our dataset. "
            "Domain pretraining gives it a vocabulary and representations tuned to clinical "
            "and self-disclosure language — the best precision on the depressed class."
        ),
        "training_data": "21 K samples",
        "accuracy": "0.96",
        "precision": "0.96",
        "recall": "0.97",
        "f1": "0.96",
        "speed": "200–400 ms",
        "explainability": "Gradient × input attribution",
    },
]

_DATASET_ROWS = [
    ("Reddit Depression (Kaggle)", "Positive (distress)", "~480 K posts", "Core positive class"),
    ("Positive subreddits (scraped)", "Negative (no distress)", "~15 K posts", "r/happy, r/aww, r/wholesomememes…"),
    ("Balanced subset", "Both", "30 K (1:1)", "Used to train all models"),
]


def _model_card(model: dict) -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(145deg, #072341 0%, #04152b 100%);
            border: 1px solid rgba(66,216,240,0.25);
            border-radius: 0.5rem;
            padding: 1.2rem 1.1rem 1rem;
            height: 100%;
        ">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                <span style="
                    background:{model["badge_color"]};
                    color:#eaf5ff;
                    font-size:0.68rem;
                    font-weight:800;
                    letter-spacing:0.05em;
                    text-transform:uppercase;
                    padding:0.2rem 0.55rem;
                    border-radius:0.2rem;
                ">{model["badge"]}</span>
            </div>
            <h3 style="margin:0 0 0.6rem; color:#ffffff; font-size:1.15rem;">{model["name"]}</h3>
            <p style="color:#bad7eb; font-size:0.88rem; line-height:1.55; margin-bottom:1rem;">
                {model["description"]}
            </p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem 1rem; font-size:0.82rem;">
                <div><span style="color:#42d8f0;">Accuracy</span><br/><b style="color:#fff;">{model["accuracy"]}</b></div>
                <div><span style="color:#42d8f0;">F1 (distress)</span><br/><b style="color:#fff;">{model["f1"]}</b></div>
                <div><span style="color:#42d8f0;">Precision</span><br/><b style="color:#fff;">{model["precision"]}</b></div>
                <div><span style="color:#42d8f0;">Recall</span><br/><b style="color:#fff;">{model["recall"]}</b></div>
                <div><span style="color:#42d8f0;">Latency</span><br/><b style="color:#fff;">{model["speed"]}</b></div>
                <div><span style="color:#42d8f0;">Training data</span><br/><b style="color:#fff;">{model["training_data"]}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_page() -> None:
    """Render the About the Models page."""
    st.markdown(
        """
        <div class="hero-banner">PROJET FINAL • BOOTCAMP DATA SCIENCE • ARTEFACT SCHOOL OF DATA</div>
        <h1 class="hero-title">About the<br/>Models</h1>
        <p class="hero-subtitle">
            Four models, one task: detect early distress signals in text.
            Each represents a different trade-off between speed, accuracy, and interpretability.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            background: rgba(255,180,0,0.08);
            border-left: 4px solid #f0a500;
            border-radius: 0 0.4rem 0.4rem 0;
            padding: 0.75rem 1rem;
            margin-bottom: 1.6rem;
            color: #f5dfa0;
            font-size: 0.9rem;
        ">
            ⚠️ <b>This is not a clinical tool.</b> It is an early-warning smoke detector —
            designed to flag potential risk, not to diagnose. Always defer to qualified
            mental health professionals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model cards ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">The models</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        _model_card(_MODELS[0])
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        _model_card(_MODELS[2])
    with col_b:
        _model_card(_MODELS[1])
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        _model_card(_MODELS[3])

    # ── Model family diagram ──────────────────────────────────────────────────
    st.markdown('<p class="section-title">Model family</p>', unsafe_allow_html=True)
    import streamlit.components.v1 as components

    components.html(
        """
        <div style="background:rgba(4,21,43,0.95); border:1px solid rgba(66,216,240,0.18);
                    border-radius:0.5rem; padding:1.5rem 1.5rem 1rem; max-width:800px; margin:0 auto;">
          <svg viewBox="0 0 800 460" xmlns="http://www.w3.org/2000/svg"
               style="width:100%; height:auto; display:block;">
            <defs>
              <marker id="mhsd-arrow" markerWidth="10" markerHeight="7"
                      refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6a8099"/>
              </marker>
            </defs>

            <!-- BERT -->
            <rect x="300" y="20" width="200" height="65" rx="10"
                  fill="rgba(40,55,70,0.9)" stroke="#8a9ab0" stroke-width="1.5"/>
            <text x="400" y="48" text-anchor="middle" fill="#ffffff"
                  font-family="sans-serif" font-size="16" font-weight="700">BERT</text>
            <text x="400" y="67" text-anchor="middle" fill="#aabbcc"
                  font-family="sans-serif" font-size="12">Bidirectional transformer</text>

            <!-- Arrow: BERT → DistilBERT -->
            <line x1="358" y1="85" x2="163" y2="193"
                  stroke="#6a8099" stroke-width="1.5" marker-end="url(#mhsd-arrow)"/>
            <text x="242" y="128" text-anchor="middle" fill="#8a9ab0"
                  font-family="sans-serif" font-size="11">distillation</text>

            <!-- Arrow: BERT → RoBERTa -->
            <line x1="412" y1="85" x2="412" y2="193"
                  stroke="#6a8099" stroke-width="1.5" marker-end="url(#mhsd-arrow)"/>
            <text x="432" y="128" text-anchor="start" fill="#8a9ab0"
                  font-family="sans-serif" font-size="11">more data,</text>
            <text x="432" y="143" text-anchor="start" fill="#8a9ab0"
                  font-family="sans-serif" font-size="11">better training</text>

            <!-- DistilBERT -->
            <rect x="45" y="195" width="220" height="75" rx="10"
                  fill="rgba(14,122,92,0.35)" stroke="#0e7a5c" stroke-width="2"/>
            <text x="155" y="226" text-anchor="middle" fill="#4eeebb"
                  font-family="sans-serif" font-size="15" font-weight="700">DistilBERT</text>
            <text x="155" y="248" text-anchor="middle" fill="#9ed8c0"
                  font-family="sans-serif" font-size="12">Smaller, faster BERT</text>

            <!-- RoBERTa -->
            <rect x="305" y="195" width="210" height="75" rx="10"
                  fill="rgba(26,110,142,0.45)" stroke="#1a6e8e" stroke-width="2"/>
            <text x="410" y="226" text-anchor="middle" fill="#42d8f0"
                  font-family="sans-serif" font-size="15" font-weight="700">RoBERTa</text>
            <text x="410" y="248" text-anchor="middle" fill="#9ad4e8"
                  font-family="sans-serif" font-size="12">Robustly optimized BERT</text>

            <!-- Arrow: RoBERTa → MentalRoBERTa -->
            <line x1="478" y1="262" x2="576" y2="328"
                  stroke="#6a8099" stroke-width="1.5" marker-end="url(#mhsd-arrow)"/>
            <text x="554" y="287" text-anchor="middle" fill="#8a9ab0"
                  font-family="sans-serif" font-size="11">domain pretraining</text>

            <!-- MentalRoBERTa -->
            <rect x="465" y="330" width="290" height="75" rx="10"
                  fill="rgba(107,63,160,0.45)" stroke="#6b3fa0" stroke-width="2"/>
            <text x="610" y="361" text-anchor="middle" fill="#c49ae0"
                  font-family="sans-serif" font-size="15" font-weight="700">MentalRoBERTa</text>
            <text x="610" y="383" text-anchor="middle" fill="#c8b0e0"
                  font-family="sans-serif" font-size="12">Reddit mental health text</text>

            <!-- Legend -->
            <rect x="30" y="428" width="13" height="13" rx="2"
                  fill="rgba(107,63,160,0.6)" stroke="#6b3fa0" stroke-width="1.5"/>
            <text x="50" y="440" fill="#8a9ab0" font-family="sans-serif" font-size="11">Domain adaptation</text>
            <rect x="205" y="428" width="13" height="13" rx="2"
                  fill="rgba(14,122,92,0.5)" stroke="#0e7a5c" stroke-width="1.5"/>
            <text x="225" y="440" fill="#8a9ab0" font-family="sans-serif" font-size="11">Knowledge distillation</text>
            <rect x="410" y="428" width="13" height="13" rx="2"
                  fill="rgba(40,55,70,0.9)" stroke="#8a9ab0" stroke-width="1.5"/>
            <text x="430" y="440" fill="#8a9ab0" font-family="sans-serif" font-size="11">Training improvements</text>
          </svg>
        </div>
        """,
        height=520,
    )

    # ── Performance comparison ────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Performance comparison — depressed class</p>', unsafe_allow_html=True)

    import pandas as pd

    df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "XGBoost", "DistilBERT", "MentalRoBERTa"],
            "Accuracy": [0.93, 0.91, 0.96, 0.96],
            "Precision": [0.93, 0.73, 0.94, 0.96],
            "Recall": [0.92, 0.87, 0.97, 0.97],
            "F1": [0.93, 0.80, 0.96, 0.96],
            "Training samples": ["30 K", "30 K", "30 K", "21 K"],
            "Latency": ["< 5 ms", "< 10 ms", "200–400 ms", "200–400 ms"],
        }
    ).set_index("Model")

    st.dataframe(df, use_container_width=True)
    st.caption(
        "All metrics on the held-out test set (15 % of balanced dataset, 30 K rows, 1:1 class ratio). "
        "Metrics shown for the **depressed class (label = 1)**."
    )

    # ── Training data ─────────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Training data</p>', unsafe_allow_html=True)

    df_data = pd.DataFrame(
        _DATASET_ROWS,
        columns=["Dataset", "Class", "Size", "Notes"],
    )
    st.dataframe(df_data, use_container_width=True, hide_index=True)

    # ── Key insights ──────────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Key insights</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div style="background:rgba(8,28,52,0.7); border:1px solid rgba(66,216,240,0.2);
                        border-radius:0.4rem; padding:1rem; font-size:0.88rem; color:#bad7eb;">
                <b style="color:#42d8f0;">Why domain pretraining matters</b><br/><br/>
                MentalRoBERTa was pretrained on mental health corpora before fine-tuning.
                This gives it vocabulary and representations tuned to clinical and
                self-disclosure language — resulting in the best precision (0.84 vs 0.75 for
                standard DistilBERT on the old dataset), meaning fewer false alarms.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="background:rgba(8,28,52,0.7); border:1px solid rgba(66,216,240,0.2);
                        border-radius:0.4rem; padding:1rem; font-size:0.88rem; color:#bad7eb;">
                <b style="color:#42d8f0;">Context vs keywords</b><br/><br/>
                LR predicts "I'm depressed" → 98 % (keyword overfit on <i>depressed</i>).
                DistilBERT predicts the same phrase → 0.4 % (context-aware: casual statement,
                not clinical distress). But both correctly score "I want to kill myself" → > 98 %.
                Transformers understand what you mean, not just what you say.
            </div>
            """,
            unsafe_allow_html=True,
        )
