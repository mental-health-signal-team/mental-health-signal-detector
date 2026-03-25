import importlib

import requests
import streamlit as st

from src.dashboard.examples import render_examples

GoogleTranslator = None
TRANSLATION_SUCCESS_NOTE = "Input was auto-translated to English before analysis."
MODEL_OPTIONS = ["lr", "xgboost", "distilbert", "mental_roberta"]
MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "xgboost": "XGBoost",
    "distilbert": "DistilBERT",
    "mental_roberta": "MentalRoBERTa",
}
DEMO_SENTENCES = {
    "Example 1": "I feel hopeless every day.",
    "Example 2": "I feel great about my life.",
    "Example 3": "I had lunch, answered emails, and finished my tasks.",
}


def _render_demo_sentence_picker(text_key: str, key_prefix: str) -> None:
    """Render controls to copy one of the predefined demo sentences into a text area."""
    selected_demo = st.selectbox(
        "You can choose a prepared sentence",
        list(DEMO_SENTENCES.keys()),
        key=f"{key_prefix}_demo_select",
    )
    if st.button("Use selected demo sentence", key=f"{key_prefix}_demo_apply"):
        st.session_state[text_key] = DEMO_SENTENCES[selected_demo]


def _get_google_translator():
    """Return GoogleTranslator class if deep_translator is installed, else None."""
    global GoogleTranslator
    if GoogleTranslator is not None:
        return GoogleTranslator

    try:
        deep_translator_module = importlib.import_module("deep_translator")
        GoogleTranslator = deep_translator_module.GoogleTranslator
        return GoogleTranslator
    except ImportError:
        return None


def _translate_to_english(text: str) -> tuple[str, str | None]:
    """Translate text to English when possible, returning translated text and optional note."""
    cleaned = text.strip()
    if not cleaned:
        return text, None

    translator_cls = _get_google_translator()
    if translator_cls is None:
        return text, "Translator unavailable (`deep-translator` not installed). Original text was used."

    try:
        translated = translator_cls(source="auto", target="en").translate(cleaned)
    except Exception:
        return text, "Automatic translation failed. Original text was used."

    if not translated:
        return text, "Automatic translation returned empty output. Original text was used."

    if translated.strip().lower() != cleaned.lower():
        return translated, TRANSLATION_SUCCESS_NOTE

    return text, None


def _render_translation_feedback(translation_note: str | None) -> None:
    """Render translation info and caption when translation was applied."""
    if not translation_note:
        return

    st.info(translation_note)
    if translation_note == TRANSLATION_SUCCESS_NOTE:
        st.caption("Prediction is based on translated English text")


def _render_hero(mode: str) -> None:
    """Render page hero with a visual style matching the project poster."""
    if mode == "prediction":
        subtitle = "NLP-powered detection of early mental distress signals."
    elif mode == "board":
        subtitle = "Compare predictions from all 4 models side-by-side — label and confidence for each."
    else:
        subtitle = "Explore which words drive each prediction via token-level attribution scores."

    st.markdown(
        """
        <div class="hero-banner">PROJET FINAL • BOOTCAMP DATA SCIENCE • ARTEFACT SCHOOL OF DATA</div>
        <h1 class="hero-title">Mental Health<br/>Signal Detector</h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<p class='hero-subtitle'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chip-row">
            <div class="chip">Logistic Regression · XGBoost · DistilBERT · MentalRoberta</div>
            <div class="chip">FastAPI · Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _probability_band_from_probability(probability: float) -> str:
    """Map depression probability to display bands aligned with a 0.5 decision threshold."""
    if probability < 0.50:
        return "Below 50%"
    if probability < 0.70:
        return "50-70%"
    if probability < 0.90:
        return "70-90%"
    return "90-100%"


def _no_distress_band_from_confidence(no_distress_confidence: float) -> str:
    """Map no-distress confidence to confidence bands for label=0 display."""
    if no_distress_confidence >= 0.90:
        return "90-100%"
    if no_distress_confidence >= 0.70:
        return "70-90%"
    if no_distress_confidence >= 0.50:
        return "50-70%"
    return "Below 50%"


def render_risk_message(label: int, probability: float) -> None:
    """Render risk message with colors based on predicted label and probability band."""
    if label == 0:
        no_distress_confidence = 1.0 - probability
        confidence_band = _no_distress_band_from_confidence(no_distress_confidence)

        if confidence_band == "90-100%":
            st.success("No distress detected. You are probably fine.")
        else:
            st.warning("No distress detected but you should still be careful.")
        return

    distress_band = _probability_band_from_probability(probability)

    if distress_band == "90-100%":
        st.error("Distress detected. You have a high depression risk.")
    else:
        st.warning("Distress detected. Not entirely confident but you might be at risk.")


def render_prediction_page(api_url: str) -> None:
    """Render the default text prediction page."""

    _render_hero("prediction")
    st.markdown(
        """
        <div style="background:rgba(255,180,0,0.08);border-left:4px solid #f0a500;
                    border-radius:0 0.4rem 0.4rem 0;padding:0.75rem 1rem;
                    margin-bottom:1.2rem;color:#f5dfa0;font-size:0.9rem;">
            ⚠️ <b>This tool is not a medical diagnosis system.</b>
            It is an early-warning smoke detector — not a clinical diagnostic tool.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_examples(session_key="predict_text")

    st.markdown('<p class="section-title">Input text</p>', unsafe_allow_html=True)

    text_input = st.text_area("Input Text", height=200, key="predict_text")
    st.markdown('<p class="section-title">Model selection</p>', unsafe_allow_html=True)
    model_type = st.selectbox("Select Model", MODEL_OPTIONS, format_func=lambda x: MODEL_DISPLAY_NAMES[x], key="predict_model")

    if st.button("Predict", key="predict_button", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return

        text_for_model, translation_note = _translate_to_english(text_input)
        _render_translation_feedback(translation_note)

        with st.spinner("Analyzing... (first request may take up to 30s to wake the server and load the model)"):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json={"text": text_for_model, "model_type": model_type},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Error: {exc}")
                return

        label = int(result["label"])
        probability = float(result["probability"])

        if label == 1:
            st.error("Distress signal detected")
            confidence = probability
            st.metric("Confidence (distress)", f"{confidence:.0%}")
            st.progress(confidence)
        else:
            st.success("No distress signal detected")
            confidence = 1 - probability
            st.metric("Confidence (no distress)", f"{confidence:.0%}")
            st.progress(confidence)

        render_risk_message(label, probability)


def render_word_importance_page(api_url: str) -> None:
    """Render the explainability page backed by the deployed API."""
    _render_hero("explain")
    render_examples(session_key="explain_sentence")
    st.markdown('<p class="section-title">Explainability sentence</p>', unsafe_allow_html=True)
    text_input = st.text_area("Sentence", height=180, key="explain_sentence")
    st.markdown('<p class="section-title">Model selection</p>', unsafe_allow_html=True)
    model_type = st.selectbox("Explain with model", MODEL_OPTIONS, format_func=lambda x: MODEL_DISPLAY_NAMES[x], key="explain_model")
    st.caption("Logistic Regression and XGBoost use TF-IDF-based word scoring. DistilBERT and MentalRoBERTa use gradient-based token attributions.")
    st.markdown('<p class="section-title">Sensitivity</p>', unsafe_allow_html=True)
    threshold = st.slider(
        "Word importance threshold",
        min_value=-2.0,
        max_value=2.0,
        value=0.05,
        step=0.001,
        key="explain_threshold",
    )
    max_tokens = 40

    if st.button("Show word importance", key="predict_with_details", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter a sentence to analyze.")
            return

        text_for_model, translation_note = _translate_to_english(text_input)
        _render_translation_feedback(translation_note)

        payload = {
            "text": text_for_model,
            "model_type": model_type,
            "threshold": threshold,
            "max_tokens": max_tokens,
        }

        with st.spinner("Analyzing token contributions..."):
            try:
                response = requests.post(f"{api_url}/explain", json=payload, timeout=180)
                if not response.ok:
                    st.error(f"Error while requesting explanation: {response.text or f'HTTP {response.status_code}'}")
                    return
                result = response.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Error while requesting explanation: {exc}")
                return

        st.markdown("### Highlighted sentence")
        st.markdown(
            (
                "<p style='color:#bad7eb;'><b style='color:#eaf5ff;'>Legend:</b> "
                '<span style="color:#4eeebb;font-weight:700;">green</span> Positive words, '
                '<span style="color:#ff6b6b;font-weight:700;">red</span> Negative words, '
                '<span style="color:#eaf5ff;background:#1a2f4a;padding:0 4px;border-radius:2px;">white</span> Neutral words</p>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                '<div style="padding: 1rem; border-radius: 0.5rem; background:#111; '
                'line-height:1.8; font-size:1.1rem;">'
                f"{result['colored_html']}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        word_importance = result.get("word_importance", {})
        if word_importance:
            top_items = sorted(word_importance.items(), key=lambda item: abs(float(item[1])), reverse=True)[:10]
            st.markdown("### Top influential words")
            if top_items:
                st.table([{"word": token, "importance": round(float(value), 4)} for token, value in top_items])
            else:
                st.info("No single-word tokens met the current threshold. Try lowering sensitivity.")


def render_models_board_page(api_url: str) -> None:
    """Render a single board with predictions from all models."""
    _render_hero("board")
    st.markdown(
        '<p style="color:var(--ink-200);font-size:0.9rem;margin-bottom:1rem;">'
        "Enter any text below to run it through all 4 models simultaneously. "
        "Results appear as a table showing each model's verdict and confidence score."
        "</p>",
        unsafe_allow_html=True,
    )
    render_examples(session_key="board_text")
    st.markdown('<p class="section-title">Input text</p>', unsafe_allow_html=True)

    text_input = st.text_area("Input Text", height=200, key="board_text")

    if st.button("Compare all models", key="board_compare_button", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return

        text_for_model, translation_note = _translate_to_english(text_input)
        _render_translation_feedback(translation_note)

        results = []
        errors = []

        with st.spinner("Running predictions for all models..."):
            for model_type in MODEL_OPTIONS:
                try:
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"text": text_for_model, "model_type": model_type},
                        timeout=60,
                    )
                    response.raise_for_status()
                    result = response.json()

                    label = int(result["label"])
                    probability = float(result["probability"])
                    distress_probability = probability if label == 1 else 1 - probability

                    results.append(
                        {
                            "model_type": model_type,
                            "model_name": MODEL_DISPLAY_NAMES[model_type],
                            "label": "Distress" if label == 1 else "No Distress",
                            "confidence": distress_probability,
                        }
                    )
                except requests.exceptions.RequestException as exc:
                    errors.append(f"{MODEL_DISPLAY_NAMES[model_type]}: {exc}")

        if results:
            summary_rows = []
            for model_result in results:
                confidence = float(model_result["confidence"])
                summary_rows.append(
                    {
                        "Model": model_result["model_name"],
                        "Label": model_result["label"],
                        "Confidence": f"{confidence:.0%}",
                    }
                )

            st.markdown("### Prediction summary")
            st.table(summary_rows)

        if errors:
            st.warning("Some models could not return a prediction.")
            for item in errors:
                st.caption(item)
