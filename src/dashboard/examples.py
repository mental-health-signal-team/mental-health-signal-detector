import random

import streamlit as st

_EXAMPLES_DISTRESS = [
    "I haven't left my room in days. Everything feels pointless and I'm so exhausted all the time.",
    "I don't know how much longer I can keep going like this. Nobody understands what I'm going through.",
    "Sometimes I think everyone would just be better off without me around.",
    "I've been feeling completely numb for weeks. I can't sleep, I can't eat, I just exist.",
    "There's no point in trying anymore. I've failed at everything and I'm so tired of fighting.",
]

_EXAMPLES_MIXED = [
    "I'm usually a happy person, but today I'm really sad and I want to kill myself.",
    "Life has been good lately, but sometimes when I'm alone at night the dark thoughts come back and I wonder if it's worth continuing.",
    "I love my family and friends but lately I've been feeling empty inside, like nothing I do actually matters and I just want it all to stop.",
    "Most days I manage fine, but this week has been rough — I keep thinking about how much easier it would be if I just wasn't here.",
    """I had a great morning and genuinely laughed for the first time in a while, but by the evening
        I was back to feeling hopeless and thinking about ending things.""",
]

_EXAMPLES_POSITIVE = [
    "Just got back from an amazing hike, the views were incredible and I feel so refreshed!",
    "Anyone have good book recommendations? Looking for something fun to read this weekend.",
    "Finally finished the project I've been working on for months. Really happy with how it turned out.",
    "Made homemade pasta for the first time today, turned out pretty good actually.",
    "Had the best time catching up with old friends last night — feeling so grateful for the people in my life.",
]


def render_examples(session_key: str = "predict_text") -> None:
    """Render three buttons that load a random example into the text input."""
    st.markdown('<p class="section-title">Try an example</p>', unsafe_allow_html=True)
    col_distress, col_mixed, col_positive = st.columns(3)

    with col_distress:
        if st.button("Distress example", key="ex_distress", use_container_width=True):
            st.session_state[session_key] = random.choice(_EXAMPLES_DISTRESS)

    with col_mixed:
        if st.button("Mixed example", key="ex_mixed", use_container_width=True):
            st.session_state[session_key] = random.choice(_EXAMPLES_MIXED)

    with col_positive:
        if st.button("Positive example", key="ex_positive", use_container_width=True):
            st.session_state[session_key] = random.choice(_EXAMPLES_POSITIVE)
