from omniopd.analysis.transfer import bag_of_words, bow_cosine, set_jaccard


def test_bow_cosine_uses_squared_count_norms():
    left = bag_of_words("apple apple fridge")
    right = bag_of_words("apple fridge fridge")
    # dot=4, both norms=sqrt(5)
    assert abs(bow_cosine(left, right) - 0.8) < 1e-12
    assert abs(bow_cosine(left, left) - 1.0) < 1e-12


def test_empty_jaccard_is_explicitly_missing():
    assert set_jaccard(set(), set()) is None
    assert set_jaccard({"a"}, {"a", "b"}) == 0.5
