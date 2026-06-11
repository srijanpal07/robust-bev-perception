"""Model components — Phase 1 (single-agent) and Phase 2 (multi-agent) exports."""

from src.models.temporal import TemporalVelocityPredictor

# Phase 1 — single-agent pipeline
from src.models.phase1_model import Phase1Model
from src.models.encoders import ResNet18BEVEncoderWithFeatures, CropEncoder
from src.models.gating import ModalityGating
from src.models.heads import TrajectoryHead

# Phase 2 — multi-agent pipeline
from src.models.encoders import RoIAgentEncoder
from src.models.gating import MultiAgentModalityGating
from src.models.interaction import AgentInteractionModule

__all__ = [
    # Phase 1
    "Phase1Model",
    "ResNet18BEVEncoderWithFeatures",
    "CropEncoder",
    "ModalityGating",
    "TrajectoryHead",
    # Phase 2
    "RoIAgentEncoder",
    "MultiAgentModalityGating",
    "AgentInteractionModule",
    # Legacy
    "TemporalVelocityPredictor",
]
