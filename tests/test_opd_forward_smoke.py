from omniopd.opd_adapter import describe_supervision_span, extract_strict_action_tokens


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


class _LenientTokenizer:
    eos_token_id = 99
    pad_token_id = 98
    all_special_ids = [98, 99]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(
            {
                1: "Action:",
                2: " look",
                3: "\n",
                4: "<think>",
                5: "I give up.",
                98: "",
                99: "<|im_end|>",
            }[token]
            for token in token_ids
        )


def test_lenient_turn_without_an_action_line_supervises_what_it_emitted():
    ids, text, action = extract_strict_action_tokens(
        _LenientTokenizer(), [5, 99, 98, 98], ["look"], require_admissible=False
    )
    assert ids == [5]
    assert text == "I give up."
    assert action is None
    assert describe_supervision_span(ids, text) == "whole_response"


def test_lenient_terminator_only_turn_supervises_that_single_token():
    ids, text, action = extract_strict_action_tokens(
        _LenientTokenizer(), [99, 98, 98], ["look"], require_admissible=False
    )
    assert ids == [99]
    assert action is None
    assert describe_supervision_span(ids, text) == "whole_response"


def test_lenient_padding_only_response_leaves_no_supervision():
    ids, text, action = extract_strict_action_tokens(
        _LenientTokenizer(), [98, 98, 98], ["look"], require_admissible=False
    )
    assert ids == []
    assert text == ""
    assert action is None
    assert describe_supervision_span(ids, text) == "empty"
