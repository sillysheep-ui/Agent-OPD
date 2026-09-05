from __future__ import annotations

from typing import Any, Sequence

from ..schema import Message
from ..tokenization import encode_final_assistant_content


def admissible_action_log_scores(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Message],
    admissible_actions: Sequence[str],
    *,
    device: Any | None = None,
) -> list[float]:
    """Score complete admissible action sequences under the training template.

    Each score is the sum of next-token log probabilities in the final
    assistant content (excluding template terminators). This implements the
    paper's sequence score without manually concatenating action tokens.
    """

    import torch
    import torch.nn.functional as F

    if not admissible_actions:
        raise ValueError("an admissible action set cannot be empty")
    examples = [
        encode_final_assistant_content(
            tokenizer,
            [*messages, {"role": "assistant", "content": f"Action: {action}"}],
            enable_thinking=False,
        )
        for action in admissible_actions
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id must be defined")
    width = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    target_mask = torch.zeros((len(examples), width), dtype=torch.float32)
    for index, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[index, :length] = torch.tensor(example.input_ids)
        attention_mask[index, :length] = 1
        target_mask[index, :length] = torch.tensor(example.target_token_mask)
    if device is None:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_mask = target_mask.to(device)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        selected = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        scores = (selected * target_mask[:, 1:]).sum(dim=1)
    return [float(value) for value in scores.cpu()]
