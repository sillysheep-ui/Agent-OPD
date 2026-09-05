from omniopd.merge import records_sha256, stable_record_id


def test_record_hash_is_order_invariant_after_canonical_sort(tmp_path=None):
    # Use direct record hashing so this test remains dependency-free in the
    # lightweight runner (pytest can separately exercise filesystem paths).
    records_a = [
        {"state_hash": "b", "group": "A1", "value": 2},
        {"state_hash": "a", "group": "A1", "value": 1},
    ]
    records_b = list(reversed(records_a))
    def key(row):
        return f"{row['group']}:{row['state_hash']}:"

    assert records_sha256(sorted(records_a, key=key)) == records_sha256(sorted(records_b, key=key))


def test_multi_teacher_samples_and_checkpoints_have_distinct_merge_ids():
    first = {
        "state_hash": "state",
        "teacher_sample_index": 0,
        "checkpoint": "before",
    }
    second = {**first, "teacher_sample_index": 1}
    after = {**first, "checkpoint": "after"}
    assert len({stable_record_id(first), stable_record_id(second), stable_record_id(after)}) == 3
