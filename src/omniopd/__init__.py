"""Agent OmniOPD: protocol-first black-box agent distillation."""

from .schema import ActionSample, AgentState, CorrectionRecord, RolloutTurn

__all__ = ["ActionSample", "AgentState", "CorrectionRecord", "RolloutTurn"]
__version__ = "0.1.0"
