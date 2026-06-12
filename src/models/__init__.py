"""Model components — Phase 1 (single-agent) and Phase 2 (multi-agent) exports."""

from src.models.phase1_model import Phase1Model

# Phase 2 — multi-agent pipeline (stubs, ready to wire)
from src.models.encoders import ResNet18BEVEncoderWithFeatures, RoIAgentEncoder
from src.models.gating import MultiAgentModalityGating
from src.models.interaction import AgentInteractionModule

__all__ = [
    # Phase 1
    "Phase1Model",
    # Phase 2
    "ResNet18BEVEncoderWithFeatures",
    "RoIAgentEncoder",
    "MultiAgentModalityGating",
    "AgentInteractionModule",
]
