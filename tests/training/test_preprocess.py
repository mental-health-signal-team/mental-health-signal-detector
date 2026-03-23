from src.training.preprocess import _normalize_text, preprocess_text


def test_normalize_text():
    """
    Test the normalization stage of preprocess_text by checking if it correctly normalizes
    URLs, user mentions, subreddit mentions, hashtags, emojis, repeated punctuation,
    and contractions.
    """
    text = "See https://example.com u/tester in r/python #Hope :) 😢"
    text2 = "HELP me at 10:30pm... Why??? no!!! 42 soooo"
    text3 = "Tom &amp; Jerry can&#39;t sleep"

    cleaned = _normalize_text(text)

    assert "urltoken" in cleaned
    assert "usertoken" in cleaned
    assert "subreddittoken" in cleaned
    assert "hashtaghope" in cleaned
    assert "emojipositive" in cleaned
    assert "emojitoken" in cleaned

    cleaned2 = _normalize_text(text2)

    assert "allcapstoken" in cleaned2
    assert "timetoken" in cleaned2
    assert "threedots" in cleaned2
    assert "questionrepeat" in cleaned2
    assert "exclamrepeat" in cleaned2
    assert "numtoken" in cleaned2
    assert "soooo" not in cleaned2
    assert "soo" in cleaned2

    cleaned3 = _normalize_text(text3)

    assert "can not" in cleaned3
    assert "&amp;" not in cleaned3
    assert "&#39;" not in cleaned3
    assert "&" not in cleaned3


def test_preprocess_text():
    """
    Test the preprocess_text function by checking if it correctly tokenizes,
    lowercases, handles punctuation according to options, and can include
    normalization in a single call.
    """
    text = "Cats are RUNNING, and dogs! Not sleeping?"
    text2 = "I am at home."
    text3 = "See https://example.com"

    cleaned = preprocess_text(
        text,
        remove_stopwords=False,
        remove_punctuation=True,
        lemmatize=False,
        normalize=False,
    )
    assert cleaned == "cats are running and dogs ! not sleeping ?"

    cleaned2 = preprocess_text(
        text2,
        remove_stopwords=False,
        remove_punctuation=False,
        lemmatize=False,
        normalize=False,
    )
    assert cleaned2 == "i am at home ."

    cleaned3 = preprocess_text(text3, remove_stopwords=False, remove_punctuation=True, lemmatize=False)
    assert cleaned3 == "see urltoken"
