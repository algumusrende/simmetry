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


# hamming_str
def test_hamming_str_identical():
    assert similarity("abc", "abc", "hamming_str") == pytest.approx(1.0)


def test_hamming_str_all_different():
    assert similarity("abc", "xyz", "hamming_str") == pytest.approx(0.0)


def test_hamming_str_partial():
    from simmetry.strings import hamming_str
    assert hamming_str("abcd", "abXd") == pytest.approx(3 / 4)


def test_hamming_str_empty():
    from simmetry.strings import hamming_str
    assert hamming_str("", "") == pytest.approx(1.0)


def test_hamming_str_length_mismatch():
    from simmetry.strings import hamming_str
    with pytest.raises(ValueError, match="same length"):
        hamming_str("ab", "abc")


# bm25
def test_bm25_identical():
    assert similarity("hello world", "hello world", "bm25") == pytest.approx(1.0)


def test_bm25_no_overlap():
    assert similarity("foo bar", "baz qux", "bm25") == pytest.approx(0.0)


def test_bm25_partial_match():
    s = similarity("hello world", "hello there", "bm25")
    assert 0.0 < s < 1.0


def test_bm25_empty_both():
    from simmetry.strings import bm25
    assert bm25("", "") == pytest.approx(1.0)


def test_bm25_one_empty():
    from simmetry.strings import bm25
    assert bm25("hello", "") == pytest.approx(0.0)
    assert bm25("", "hello") == pytest.approx(0.0)


def test_bm25_asymmetric():
    from simmetry.strings import bm25
    # bm25 is not symmetric by design
    ab = bm25("hello", "hello world")
    ba = bm25("hello world", "hello")
    assert ab != pytest.approx(ba)
