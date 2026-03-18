import html
import re
import string


def preprocess_text(text: str, model_type="lr") -> str:
    """
    Preprocess the text by removing HTML entities, URLs, subreddit mentions,
    and normalizing punctuation and case for logistic regression.
    - For logistic regression, we replace "..."
    with "three_dots" to preserve it as a feature,
    and we remove all punctuation except for "!", "?", and "'".
    - We also reduce repeated characters to a maximum of two occurrences
    (e.g., "so happyyyyy" becomes "so happyy") to help with normalization.
    """
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"r/[\w-]+", "", text)
    if model_type == "lr":
        text = text.replace("...", "three_dots")
        punctuation_to_remove = string.punctuation.replace("!", "").replace("?", "").replace("'", "")
        text = re.sub(f"[{re.escape(punctuation_to_remove)}]", "", text)
        text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
