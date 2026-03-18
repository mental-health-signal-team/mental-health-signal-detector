import src.training.preprocess as preprocess_module
from src.training.preprocess import clean_text, preprocess_text


def test_clean_text():
    """
    Test the clean_text function by checking if it correctly normalizes
    URLs, user mentions, subreddit mentions, hashtags, emojis, repeated punctuation,
    and contractions.
    """
    text = "See https://example.com u/tester in r/python #Hope :) 😢"
    text2 = "HELP me at 10:30pm... Why??? no!!! 42 soooo"
    text3 = "Tom &amp; Jerry can&#39;t sleep"

    cleaned = clean_text(text)

    # Underscores are removed by punctuation filtering, so tokens become compact.
    assert "urltoken" in cleaned
    assert "usertoken" in cleaned
    assert "subreddittoken" in cleaned
    assert "hashtaghope" in cleaned
    assert "emojipositive" in cleaned
    assert "emojitoken" in cleaned

    cleaned2 = clean_text(text2)

    assert "allcapstoken" in cleaned2
    assert "timetoken" in cleaned2
    assert "threedots" in cleaned2
    assert "questionrepeat" in cleaned2
    assert "exclamrepeat" in cleaned2
    assert "numtoken" in cleaned2
    assert "soooo" not in cleaned2
    assert "soo" in cleaned2

    cleaned3 = clean_text(text3)

    assert "can not" in cleaned3
    assert "&amp;" not in cleaned3
    assert "&#39;" not in cleaned3
    assert "&" not in cleaned3


def test_preprocess_text(monkeypatch):
    """
    Test the preprocess_text function by checking if it correctly tokenizes,
    lowercases, removes stop words and punctuation, and lemmatizes the text.
    We use monkeypatching to replace the word_tokenize, pos_tag, and lemmat"""
    text = "Cats are RUNNING, and dogs! Not sleeping?"
    text2 = "I am at home."
    text3 = "played better cars"

    def fake_word_tokenize(value):
        mapping = {
            text: [
                "Cats",
                "are",
                "RUNNING",
                ",",
                "and",
                "dogs",
                "!",
                "Not",
                "sleeping",
                "?",
            ],
            text2: ["I", "am", "at", "home", "."],
            text3: ["played", "better", "cars"],
        }
        return mapping[value]

    class FakeLemmatizer:
        def lemmatize(self, token, pos):
            lemma_map = {
                ("cats", "n"): "cat",
                ("running", "v"): "run",
                ("dogs", "n"): "dog",
                ("sleeping", "v"): "sleep",
                ("played", "v"): "play",
                ("better", "a"): "good",
                ("cars", "n"): "car",
            }
            return lemma_map.get((token, pos), token)

    def fake_pos_tag(tokens):
        tag_map = {
            "cats": "NNS",
            "running": "VBG",
            "dogs": "NNS",
            "not": "RB",
            "sleeping": "VBG",
            "played": "VBD",
            "better": "JJR",
            "cars": "NNS",
            "!": ".",
            "?": ".",
            "i": "PRP",
            "am": "VBP",
            "at": "IN",
            "home": "NN",
            ".": ".",
        }
        return [(token, tag_map.get(token, "NN")) for token in tokens]

    monkeypatch.setattr(preprocess_module, "word_tokenize", fake_word_tokenize)
    monkeypatch.setattr(preprocess_module, "_get_stop_words", lambda: {"are", "and", "i", "am", "at"})
    monkeypatch.setattr(preprocess_module.nltk, "pos_tag", fake_pos_tag)
    monkeypatch.setattr(preprocess_module, "_LEMMATIZER", FakeLemmatizer())

    cleaned = preprocess_text(text)
    assert "cat" in cleaned
    assert "run" in cleaned
    assert "dog" in cleaned
    assert "not" in cleaned
    assert "sleep" in cleaned
    assert "!" in cleaned
    assert "?" in cleaned
    assert "are" not in cleaned
    assert "and" not in cleaned

    cleaned2 = preprocess_text(text2, remove_stopwords=False, remove_punctuation=False, lemmatize=False)
    assert cleaned2 == "i am at home ."

    cleaned3 = preprocess_text(text3, remove_stopwords=False, remove_punctuation=True, lemmatize=True)
    assert cleaned3 == "play good car"
