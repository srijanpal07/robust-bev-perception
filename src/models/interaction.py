"""Agent-agent interaction module for multi-agent trajectory forecasting.

Phase 2 — adapted from HiVT [Zhou et al., CVPR 2022].
HiVT paper: https://arxiv.org/abs/2206.10924

Standard HiVT local context attention:
    attn_{ij} = softmax(Q_i · K_j / sqrt(d_k))

Degradation-aware extension (novel contribution):
    attn_{ij} = softmax((Q_i · K_j - λ · (1 - q_j)) / sqrt(d_k))

q_j ∈ [0, 1] is the per-agent quality score from ModalityGating. Agents with
poor LiDAR coverage contribute less context to their neighbours — the first
interaction model to account for per-agent sensor quality.

λ is a learnable scalar (log-parameterised, initialised to 1.0) rather than a
fixed hyperparameter, letting the model learn how strongly to penalise degraded
context sources.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationAwareAttention(nn.Module):
    """Multi-head attention with per-agent quality-score penalty on key agents.

    Args:
        d_model: embedding dimension (must be divisible by nhead)
        nhead:   number of attention heads
    """

    def __init__(self, d_model: int = 256, nhead: int = 4):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k   = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out    = nn.Linear(d_model, d_model)
        # Learnable penalty weight — log-parameterised so λ = exp(log_λ) > 0 always
        self.log_lam = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, q_scores: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:        (B, N, d_model) — per-agent embeddings
            q_scores: (B, N, 1)       — per-agent quality scores in [0, 1]
            mask:     (B, N, N) bool  — True = ignore this (i, j) pair (padding)

        Returns:
            (B, N, d_model) — interaction-refined embeddings
        """
        B, N, _ = x.shape
        lam = self.log_lam.exp()

        def split_heads(t):
            return t.view(B, N, self.nhead, self.d_k).transpose(1, 2)

        Q = split_heads(self.q_proj(x))   # (B, h, N, d_k)
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # Scaled dot-product attention logits
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, N, N)

        # Degradation penalty on key (source) agents:
        #   attn[b, h, i, j] -= λ · (1 - q_j)
        # q_scores: (B, N, 1) → penalty: (B, 1, 1, N)
        penalty = lam * (1.0 - q_scores.squeeze(-1))  # (B, N)
        attn    = attn - penalty.unsqueeze(1).unsqueeze(2)

        if mask is not None:
            # mask: (B, N, N), True = ignore → expand over heads
            attn = attn.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, V)                               # (B, h, N, d_k)
        out  = out.transpose(1, 2).reshape(B, N, -1)              # (B, N, d_model)
        return self.out(out)


class AgentInteractionLayer(nn.Module):
    """One interaction layer: DegradationAwareAttention + FFN with pre-norm residuals."""

    def __init__(self, d_model: int = 256, nhead: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn  = DegradationAwareAttention(d_model, nhead)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q_scores: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), q_scores, mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class AgentInteractionModule(nn.Module):
    """Stacked degradation-aware interaction layers.

    Architecture credit: HiVT local context encoder [Zhou et al., CVPR 2022].
    Novel extension: DegradationAwareAttention — down-weights poorly-covered
    agents as context sources using per-agent quality scores from ModalityGating.

    Phase 2: not yet wired into the training pipeline (awaits multi-agent
    dataset and per-agent gating implementation).

    Args:
        d_model:         per-agent embedding dimension (must match temporal output)
        nhead:           attention heads
        num_layers:      number of stacked interaction layers
        dim_feedforward: FFN inner width
        dropout:         dropout rate
    """

    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AgentInteractionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, q_scores: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:        (B, N, d_model) — per-agent embeddings after temporal + gating
            q_scores: (B, N, 1)       — per-agent LiDAR quality scores
            mask:     (B, N, N) bool  — True = ignore padding agents

        Returns:
            (B, N, d_model) — interaction-refined per-agent embeddings
        """
        for layer in self.layers:
            x = layer(x, q_scores, mask)
        return x
