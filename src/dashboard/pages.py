import requests
import streamlit as st


def _render_hero(mode: str) -> None:
    """Render page hero with a visual style matching the project poster."""
    if mode == "prediction":
        subtitle = "Détection de signaux de detresse mentale par NLP."
    else:
        subtitle = "Explorez les mots qui influencent la prediction pour mieux expliquer chaque resultat au niveau token."

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
            <div class="chip">Logistic Regression · XGBoost · DistilBERT · MentalBERT</div>
            <div class="chip">FastAPI · Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_message(risk_level: str) -> None:
    """Render categorical risk level with color-coded Streamlit message."""
    if risk_level == "low":
        st.success("Depression risk: Low")
    elif risk_level == "medium":
        st.warning("Depression risk: Medium")
    else:
        st.error("Depression risk: High")


def render_prediction_page(api_url: str) -> None:
    """Render the default text prediction page."""
    _render_hero("prediction")
    st.markdown('<p class="section-title">Input text</p>', unsafe_allow_html=True)

    text_input = st.text_area("Input Text", height=200, key="predict_text")
    st.markdown('<p class="section-title">Model selection</p>', unsafe_allow_html=True)
    model_type = st.selectbox("Select Model", ["lr", "distilbert"], key="predict_model")

    if st.button("Predict", key="predict_button"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return

        with st.spinner("Analyzing... (first request may take up to 30s to wake the server)"):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json={"text": text_input, "model_type": model_type},
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

        if probability < 0.33:
            render_risk_message("low")
        elif probability < 0.66:
            render_risk_message("medium")
        else:
            render_risk_message("high")


def render_word_importance_page(api_url: str) -> None:
    """Render the explainability page backed by the deployed API."""
    _render_hero("explain")
    st.markdown('<p class="section-title">Explainability sentence</p>', unsafe_allow_html=True)

    text_input = st.text_area("Sentence", height=180, key="explain_sentence")
    st.markdown('<p class="section-title">Model selection</p>', unsafe_allow_html=True)
    model_type = st.selectbox("Explain with model", ["lr", "distilbert"], key="explain_model")
    st.caption("Choose LR for sparse token contributions or DistilBERT for gradient-based token attributions.")
    st.markdown('<p class="section-title">Sensitivity</p>', unsafe_allow_html=True)
    threshold = st.slider(
        "Word importance threshold",
        min_value=0.0,
        max_value=0.1,
        value=0.005,
        step=0.001,
        key="explain_threshold",
    )
    max_tokens = 40

    if st.button("Show word importance", key="predict_with_details"):
        if not text_input.strip():
            st.warning("Please enter a sentence to analyze.")
            return

        payload = {
            "text": text_input,
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
                "<p><b>Legend:</b> "
                '<span style="color:green">green</span> Positive words, '
                '<span style="color:red">red</span> Negative words, '
                '<span style="color:white;background:#111;padding:0 4px;">white</span> Neutral words</p>'
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
            st.table([{"word": token, "importance": round(float(value), 4)} for token, value in top_items])
