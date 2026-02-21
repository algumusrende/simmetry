from simfast import similarity


def test_haversine_similarity_range():
    # Istanbul-ish points; similarity should be less than 1 but > 0
    a = (41.0082, 28.9784)
    b = (41.015, 28.95)
    s = similarity(a, b, "haversine_km")
    assert 0.0 < s < 1.0


def test_set_jaccard_dice_overlap():
    A = {1, 2, 3}
    B = {2, 3, 4}
    assert similarity(A, B, "jaccard") == 2 / 4
    assert similarity(A, B, "overlap") == 2 / 3
    assert similarity(A, B, "dice") == (2 * 2) / (3 + 3)
