import pytest

from simmetry import similarity
from simmetry.strings import levenshtein, levenshtein_distance


def test_levenshtein_basic():
    assert similarity("kitten", "kitten", "levenshtein") == pytest.approx(1.0)
    assert similarity("", "", "levenshtein") == pytest.approx(1.0)
    assert 0.0 <= similarity("kitten", "sitting", "levenshtein") <= 1.0


def test_levenshtein_empty_vs_nonempty():
    assert levenshtein("", "abc") == pytest.approx(0.0)
    assert levenshtein("abc", "") == pytest.approx(0.0)
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "") == 3


def test_levenshtein_both_empty():
    assert levenshtein("", "") == pytest.approx(1.0)
    assert levenshtein_distance("", "") == 0


def test_jaro_winkler_basic():
    s = similarity("martha", "marhta", "jaro_winkler")
    assert 0.8 < s <= 1.0


def test_ngram_jaccard():
    s1 = similarity("hello", "hello", "ngram_jaccard")
    s2 = similarity("hello", "world", "ngram_jaccard")
    assert s1 == pytest.approx(1.0)
    assert s2 < 1.0


def test_token_jaccard_identical():
    assert similarity("hello world", "hello world", "token_jaccard") == pytest.approx(1.0)


def test_token_jaccard_no_overlap():
    assert similarity("foo bar", "baz qux", "token_jaccard") == pytest.approx(0.0)


def test_token_jaccard_partial():
    s = similarity("hello world", "hello there", "token_jaccard")
    assert 0.0 < s < 1.0


def test_token_jaccard_empty_strings():
    from simmetry.strings import token_jaccard
    assert token_jaccard("", "") == pytest.approx(1.0)


def test_token_jaccard_one_empty():
    from simmetry.strings import token_jaccard
    assert token_jaccard("", "hello") == pytest.approx(0.0)
