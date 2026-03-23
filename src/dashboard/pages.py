import requests
import streamlit as st


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
    st.title("Mental Health Signal Detector")
    st.write("Enter text to analyze for mental health signals.")

    text_input = st.text_area("Input Text", height=200, key="predict_text")
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
    st.title("Word Importance Details")
    st.write("Predict and highlight token importance inside the typed sentence.")

    text_input = st.text_area("Sentence", height=180, key="explain_sentence")
    model_type = "lr"
    st.caption("Word-level explanation currently supports the LR model.")
    threshold = st.slider(
        "Color threshold",
        min_value=0.0,
        max_value=0.05,
        value=0.005,
        step=0.001,
        key="explain_threshold",
    )
    max_tokens = 40

    if st.button("Predict with details", key="predict_with_details"):
        if not text_input.strip():
            st.warning("Please enter a sentence to analyze.")
            return

        payload = {
            "text": text_input,
            "model_type": model_type,
            "threshold": threshold,
            "max_tokens": max_tokens,
        }

        with st.spinner("Generating prediction and explanation..."):
            try:
                response = requests.post(f"{api_url}/explain", json=payload, timeout=180)
                if not response.ok:
                    st.error(f"Error while requesting explanation: {response.text or f'HTTP {response.status_code}'}")
                    return
                result = response.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Error while requesting explanation: {exc}")
                return

        label = int(result["label"])
        confidence = float(result["display_confidence"])
        confidence_label = str(result["confidence_label"])
        if label == 1:
            st.error("Distress signal detected")
        else:
            st.success("No distress signal detected")

        st.metric(f"Confidence ({confidence_label.replace('_', ' ')})", f"{confidence:.0%}")
        st.progress(confidence)

        render_risk_message(str(result["risk_level"]))
        note = result.get("note")
        if note:
            st.info(str(note))

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
