from simmetry import similarity


def test_levenshtein_basic():
    assert similarity("kitten", "kitten", "levenshtein") == 1.0
    assert similarity("", "", "levenshtein") == 1.0
    assert 0.0 <= similarity("kitten", "sitting", "levenshtein") <= 1.0


def test_jaro_winkler_basic():
    s = similarity("martha", "marhta", "jaro_winkler")
    assert 0.8 < s <= 1.0


def test_ngram_jaccard():
    s1 = similarity("hello", "hello", "ngram_jaccard")
    s2 = similarity("hello", "world", "ngram_jaccard")
    assert s1 == 1.0
    assert s2 < 1.0
