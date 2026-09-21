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


def test_reference_grammar_spellings_normalise_to_environment_commands():
    from omniopd.parser import parse_action

    admissible = (
        "heat apple 1 with microwave 1",
        "clean mug 1 with sinkbasin 1",
        "cool tomato 2 with fridge 1",
        "move soapbottle 2 to drawer 1",
    )
    assert parse_action("Action: heat apple 1 using microwave 1", admissible).canonical_action == (
        "heat apple 1 with microwave 1"
    )
    assert parse_action("Action: clean mug 1 using sinkbasin 1", admissible).canonical_action == (
        "clean mug 1 with sinkbasin 1"
    )
    assert parse_action("Action: cool tomato 2 using fridge 1", admissible).canonical_action == (
        "cool tomato 2 with fridge 1"
    )
    assert parse_action("Action: put soapbottle 2 in drawer 1", admissible).canonical_action == (
        "move soapbottle 2 to drawer 1"
    )
