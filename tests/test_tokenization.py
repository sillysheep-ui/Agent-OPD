from omniopd.tokenization import (
    encode_final_assistant_content,
    encode_final_assistant_example,
    prediction_mask,
    token_weights,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking=False):
        assert tokenize
        ids = [1]
        for message in messages:
            if message["role"] == "system":
                ids += [10]
            elif message["role"] == "user":
                ids += [20]
            elif message["role"] == "assistant":
                ids += [30, 31, 32]
        if add_generation_prompt:
            ids += [30]
        return ids


def test_target_mask_marks_assistant_content_and_alignment_shifts_right():
    encoded = encode_final_assistant_example(
        FakeTokenizer(),
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "Action: look"},
        ],
    )
    assert encoded.input_ids == [1, 10, 20, 30, 31, 32]
    assert encoded.target_token_mask == [0, 0, 0, 0, 1, 1]
    assert prediction_mask(encoded.target_token_mask) == [0, 0, 0, 1, 1]
    assert token_weights(encoded.target_token_mask, 0.5) == [0, 0, 0, 0, 0.25, 0.25]


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": super().apply_chat_template(*args, **kwargs)}


def test_transformers_mapping_chat_template_return_is_supported():
    encoded = encode_final_assistant_example(
        MappingTokenizer(),
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "Action: look"},
        ],
    )
    assert encoded.input_ids == [1, 10, 20, 30, 31, 32]


class BrokenTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking=False):
        ids = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        return [99] + ids if not add_generation_prompt and messages[-1]["role"] == "assistant" else ids


def test_template_mismatch_fails_closed():
    try:
        encode_final_assistant_example(
            BrokenTokenizer(),
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ],
        )
    except ValueError as error:
        assert "prefix mismatch" in str(error)
    else:
        raise AssertionError("broken templates must not silently concatenate tokenizations")


class ContentAwareTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        ids = [1, 10, 20]
        if add_generation_prompt:
            return ids + [30]
        if messages[-1]["role"] == "assistant":
            content = messages[-1]["content"]
            return ids + [30] + ([40, 41] if content else []) + [32]
        return ids


def test_action_score_mask_excludes_the_chat_template_terminator():
    encoded = encode_final_assistant_content(
        ContentAwareTokenizer(),
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "Action: look"},
        ],
    )
    assert encoded.input_ids == [1, 10, 20, 30, 40, 41, 32]
    assert encoded.target_token_mask == [0, 0, 0, 0, 1, 1, 0]


class QwenLikeBridgeTokenizer:
    @staticmethod
    def _render(messages, add_generation_prompt, enable_thinking):
        text = ""
        for message in messages:
            text += f"<{message['role']}>" + message["content"] + "</turn>"
        if add_generation_prompt:
            text += "<assistant>"
            if not enable_thinking:
                text += "<think>\n\n</think>\n\n"
        return text

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        text = self._render(messages, add_generation_prompt, enable_thinking)
        return [ord(character) for character in text] if tokenize else text


def test_qwen_generation_only_nonthinking_prefix_is_bridged_and_masked():
    tokenizer = QwenLikeBridgeTokenizer()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "Action: look"},
    ]
    training = encode_final_assistant_example(tokenizer, messages, enable_thinking=False)
    rendered = "".join(chr(token) for token in training.input_ids)
    assert "<assistant><think>\n\n</think>\n\nAction: look</turn>" in rendered
    prompt_length = len(training.input_ids) - training.target_length
    assert all(value == 0 for value in training.target_token_mask[:prompt_length])
    content = encode_final_assistant_content(tokenizer, messages, enable_thinking=False)
    scored = "".join(
        chr(token)
        for token, selected in zip(content.input_ids, content.target_token_mask)
        if selected
    )
    assert scored == "Action: look"
