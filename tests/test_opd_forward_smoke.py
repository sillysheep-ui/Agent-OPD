from omniopd.opd_adapter import extract_strict_action_tokens


class _Tokenizer:
    eos_token_id = 99
    all_special_ids = [98, 99]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join({1: "Action:", 2: " look", 3: "\n", 4: "<think>"}[token]
                       for token in token_ids)


def test_strict_action_only_tokens_exclude_eos_and_keep_action_line():
    ids, text, action = extract_strict_action_tokens(
        _Tokenizer(), [1, 2, 3, 99], ["look", "inventory"]
    )
    assert ids == [1, 2, 3]
    assert text == "Action: look\n"
    assert action == "look"


def test_strict_action_only_tokens_reject_incomplete_or_nonaction_response():
    for generated in ([1, 2], [1, 2, 98, 99], [4, 1, 2, 99], [1, 2, 99]):
        try:
            extract_strict_action_tokens(_Tokenizer(), generated, ["inventory"])
        except ValueError:
            continue
        raise AssertionError(f"invalid action response was accepted: {generated}")
