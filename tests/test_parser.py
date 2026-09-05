from omniopd.parser import parse_action


def test_parser_is_strict_by_default():
    result = parse_action("Action: take apple", ["take apple 1", "take apple 2"])
    assert not result.valid
    assert result.failure_reason == "not_admissible"


def test_parser_normalizes_put_grammar_only_when_exact_after_normalization():
    result = parse_action("Action: put apple 1 in fridge 1", ["move apple 1 to fridge 1"])
    assert result.valid
    assert result.canonical_action == "move apple 1 to fridge 1"


def test_parser_rejects_multiple_action_markers():
    result = parse_action("Action: look\nAction: go", ["look", "go"])
    assert not result.valid
    assert result.failure_reason == "multiple_action_markers"
