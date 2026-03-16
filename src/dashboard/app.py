"""
Dashboard Streamlit — Mental Health Signal Detector
Saisie d'un texte → score de risque + explication
"""

import os
import re

import pandas as pd
import requests
import streamlit as st

_raw_url = os.getenv("API_URL", "http://localhost:8000")
# Normaliser avant validation : un slash final (http://host:8000/) est courant
# et ne constitue pas un risque SSRF — le rejeter serait un faux positif silencieux.
_normalized_url = _raw_url.rstrip("/")
# Autorise uniquement http(s)://host:port — bloque les URLs internes arbitraires (SSRF)
if not re.fullmatch(r"https?://[\w.\-]+(:\d+)?", _normalized_url):
    API_URL = "http://localhost:8000"
else:
    API_URL = _normalized_url

st.set_page_config(
    page_title="Mental Health Signal Detector",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Mental Health Signal Detector")
st.caption("Détection de signaux de détresse mentale via NLP — Artefact School of Data")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("Paramètres")
    model_type = st.selectbox("Modèle", ["baseline", "distilbert"], index=0)
    st.info(
        "**baseline** : Logistic Regression + TF-IDF (rapide)\n\n"
        "**distilbert** : DistilBERT fine-tuned (plus précis)"
    )
    st.divider()
    st.markdown("⚠️ *Ce système n'est pas un diagnostic clinique. C'est un détecteur de signal.*")

# --- Prédiction manuelle ---
st.subheader("Analyser un texte")
text_input = st.text_area(
    "Entrez un texte à analyser",
    placeholder="Ex: I've been feeling really hopeless lately and can't find any motivation...",
    height=150,
)

show_shap = st.checkbox("Afficher l'explication SHAP", value=True, disabled=model_type != "baseline",
                        help="Disponible uniquement pour le modèle baseline")

if st.button("Analyser", type="primary", disabled=not text_input.strip()):
    with st.spinner("Analyse en cours..."):
        try:
            endpoint = "/explain" if (show_shap and model_type == "baseline") else "/predict"
            payload = {"text": text_input}
            if endpoint == "/predict":
                payload["model_type"] = model_type
            else:
                payload["n_features"] = 15

            response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()

            score = result["score_distress"]
            label = result["label"]
            detected_lang = result.get("detected_lang", "en")
            lang_label = {"fr": "Français", "en": "Anglais"}.get(detected_lang, detected_lang)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score de risque", f"{score:.1%}")
            with col2:
                if label == 1:
                    st.error("🔴 Signal de détresse")
                else:
                    st.success("🟢 Pas de détresse")
            with col3:
                st.metric("Langue détectée", lang_label)

            st.progress(score)

            if score > 0.7:
                st.warning("Score élevé : signal potentiel de détresse mentale.")
            elif score > 0.4:
                st.info("Score modéré : surveiller.")

            # --- SHAP ---
            if endpoint == "/explain" and result.get("features"):
                st.divider()
                st.subheader("Mots les plus influents (SHAP)")
                st.caption("Valeurs positives → pousse vers **détresse** · Valeurs négatives → pousse vers **non-détresse**")

                features = result["features"]
                df_shap = pd.DataFrame(features).sort_values("shap_value")

                colors = ["#d73027" if v > 0 else "#4575b4" for v in df_shap["shap_value"]]

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, max(3, len(df_shap) * 0.35)))
                bars = ax.barh(df_shap["word"], df_shap["shap_value"], color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Contribution SHAP")
                ax.set_title("Impact des mots sur le score de détresse")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l'API. Vérifiez que le serveur FastAPI est démarré.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                st.warning("Modèle **distilbert** non disponible — il doit être fine-tuné d'abord.\n\n"
                           "Lance : `python -m src.training.train --model distilbert`")
            else:
                st.error(f"Erreur API : {e}")
        except Exception as e:
            st.error(f"Erreur : {e}")

st.divider()

# --- Statut API ---
st.subheader("Statut de l'API")
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health["model_loaded"]:
        st.success("API opérationnelle — modèle chargé")
    else:
        st.warning("API démarrée mais modèle non chargé (entraîner d'abord)")
except Exception:
    st.error("API hors ligne")
