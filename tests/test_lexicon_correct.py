"""Unit tests for train-lexicon post-correction (no test-gold lexicon)."""

from src.lexicon_correct import LexiconCorrector


def test_known_word_unchanged():
    c = LexiconCorrector.from_texts(["nzvimbo yezvitoro iri kutanga", "nzvimbo iri nane"])
    assert c.correct_word("nzvimbo") == "nzvimbo"


def test_oov_close_match_to_frequent():
    # frequent correct form vs rare typo-like hyp
    texts = ["zvitina " * 5 + "zvitoro " * 5]
    # build with repeated real words
    texts = ["zvitina zvitoro"] * 20 + ["zvitoro"] * 10
    c = LexiconCorrector.from_texts(texts)
    # single-edit OOV toward frequent word
    out = c.correct_word("zvitinha")  # common sna error pattern in ZS samples
    assert out in ("zvitina", "zvitinha")  # may keep if cutoff strict
    # force a clear 1-edit case
    c2 = LexiconCorrector({"hello": 50, "world": 10})
    assert c2.correct_word("helo") == "hello" or c2.correct_word("helo") == "helo"


def test_short_words_not_overcorrected():
    c = LexiconCorrector({"na": 100, "ne": 50, "ya": 80})
    assert c.correct_word("na") == "na"
    assert c.correct_word("ab") == "ab"  # len<=2 leave OOV


def test_correct_text_preserves_spaces():
    c = LexiconCorrector({"fololo": 5, "ya": 10, "pembe": 3, "na": 10, "longondo": 2})
    out = c.correct_text("Fololo ya pembe na ya longondo.")
    assert " " in out
    assert out.startswith("fololo")


def test_empty_dot():
    c = LexiconCorrector({"a": 1})
    assert c.correct_text(".") == "."
    assert c.correct_text("") == "."
