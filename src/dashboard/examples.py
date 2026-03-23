import random

import streamlit as st

_EXAMPLES_DISTRESS = [
    "I haven't left my room in days. Everything feels pointless and I'm so exhausted all the time.",
    "I don't know how much longer I can keep going like this. Nobody understands what I'm going through.",
    "Sometimes I think everyone would just be better off without me around.",
    "I've been feeling completely numb for weeks. I can't sleep, I can't eat, I just exist.",
]

_EXAMPLES_NEUTRAL = [
    "Just got back from an amazing hike, the views were incredible and I feel so refreshed!",
    "Anyone have good book recommendations? Looking for something fun to read this weekend.",
    "Finally finished the project I've been working on for months. Really happy with how it turned out.",
    "Made homemade pasta for the first time today, turned out pretty good actually.",
]


def render_examples(session_key: str = "predict_text") -> None:
    """Render two buttons that load a random example into the text input."""
    st.markdown('<p class="section-title">Try an example</p>', unsafe_allow_html=True)
    col_distress, col_neutral = st.columns(2)

    with col_distress:
        if st.button("Distress example", key="ex_distress", use_container_width=True):
            st.session_state[session_key] = random.choice(_EXAMPLES_DISTRESS)

    with col_neutral:
        if st.button("Neutral example", key="ex_neutral", use_container_width=True):
            st.session_state[session_key] = random.choice(_EXAMPLES_NEUTRAL)
