"""veRL custom SFT dataset: train ONLY the final assistant message.
 
Route A samples contain previous Student assistant turns as state/history. Those turns
must be context-only. The default MultiTurnSFTDataset marks every assistant message
as loss-bearing, which would silently mix Student self-imitation into Teacher OPD.
 
NOTE (veRL 0.4.1 compatibility): the hook used here is `_process_message_tokens`,
which is the method MultiTurnSFTDataset.__getitem__ actually calls for each message
group. Overriding `_process_single_message` (older veRL) would be a silent no-op.
"""
from __future__ import annotations
 
from typing import Any, Dict, List, Optional, Tuple
 
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
 
 
class FinalTurnOnlySFTDataset(MultiTurnSFTDataset):
    def _process_message_tokens(
        self,
        messages: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        tokens, loss_mask, attention_mask = super()._process_message_tokens(
            messages=messages,
            start_idx=start_idx,
            end_idx=end_idx,
            is_assistant=is_assistant,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        if is_assistant:
            last_assistant = max(
                (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
                default=-1,
            )
            if start_idx != last_assistant:
                loss_mask = [0] * len(loss_mask)
        return tokens, loss_mask, attention_mask
