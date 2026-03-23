import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Mental Health Signal Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        max-width: 900px;
    }
    .metric-container {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Header
st.title("🧠 Mental Health Signal Detector")
st.markdown(
    "Analyze text for potential mental health signals using advanced machine learning models."
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_type = st.radio(
        "Select Prediction Model",
        options=["lr", "xgboost", "distilbert", "roberta"],
        format_func=lambda x: {
            "lr": "🔷 Logistic Regression (Fast)",
            "xgboost": "🟨 XGBoost (Balanced)",
            "distilbert": "🤖 DistilBERT (Advanced)",
            "roberta": "🧠 RoBERTa Base (Advanced)",
        }[x],
    )
    
    st.markdown("---")
    st.subheader("ℹ️ About Models")
    st.markdown(
        """
        - **Logistic Regression**: Fast and interpretable
        - **XGBoost**: Balanced accuracy and speed
        - **DistilBERT**: Advanced NLP model (slower)
        - **RoBERTa Base**: Advanced transformer model
        """
    )

# Main content
st.markdown("### 📝 Input Text")
text_input = st.text_area(
    "Enter text to analyze for mental health signals",
    height=150,
    placeholder="Paste or type your text here...",
)

# Health check
if st.button("🔍 Check API Health", use_container_width=True):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is online and ready")
        else:
            st.error(f"❌ API returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Cannot connect to API: {e}")

# Prediction
col1, col2 = st.columns([3, 1])

with col1:
    predict_btn = st.button("🚀 Analyze Text", use_container_width=True, type="primary")

with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if predict_btn:
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner(f"🔄 Analyzing with {model_type.upper()} model..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_input, "model_type": model_type},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                
                label = result.get("label", 0)
                probability = result.get("probability", 0)
                
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                # Result display
                if label == 1:
                    st.error(
                        f"🚨 **Distress Signal Detected**\n\n"
                        f"Confidence: {probability:.1%}",
                        icon="⚠️"
                    )
                else:
                    st.success(
                        f"✅ **No Distress Signal**\n\n"
                        f"Confidence: {1-probability:.1%}",
                        icon="👍"
                    )
                
                # Metric display
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Signal Probability",
                        f"{probability:.1%}",
                        delta=None,
                    )
                with col2:
                    st.metric(
                        "Safe Probability",
                        f"{1-probability:.1%}",
                        delta=None,
                    )
                
                # Progress bar
                st.progress(probability, text=f"Signal Strength: {probability:.1%}")
                
                st.markdown("---")
                st.info(
                    "💡 **Note**: This tool is for informational purposes only. "
                    "If you're concerned about mental health, please consult a professional."
                )
                
            except requests.exceptions.Timeout:
                st.error("❌ Request timeout. The API might be slow or offline.")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Cannot connect to API at {API_URL}. "
                    "Make sure the API service is running."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API Error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

