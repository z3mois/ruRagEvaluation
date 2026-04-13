import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict

# class DebertaTrace(nn.Module):
#     def __init__(self, base_model_name: str):
#         super().__init__()
#         self.base = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
#         hid = self.base.config.hidden_size

#         self.rel_head = nn.Linear(hid, 1)
#         self.util_head = nn.Linear(hid, 1)
#         self.adh_head = nn.Linear(hid, 1)

#     def forward(self, input_ids, attention_mask, **kwargs) -> Dict[str, torch.Tensor]:
#         out = self.base(input_ids=input_ids, attention_mask=attention_mask)
#         hs = out.last_hidden_state                      # (B, T, H)

#         logits_rel  = self.rel_head(hs).squeeze(-1)     # (B, T)
#         logits_util = self.util_head(hs).squeeze(-1)    # (B, T)
#         logits_adh  = self.adh_head(hs).squeeze(-1)     # (B, T)

#         return {
#             "logits_relevance":   logits_rel,
#             "logits_utilization": logits_util,
#             "logits_adherence":   logits_adh,
#         }
from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoModel
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "find_pruneable_heads_and_indices") and hasattr(_pu, "find_prunable_heads_and_indices"):
    _pu.find_pruneable_heads_and_indices = _pu.find_prunable_heads_and_indices

class TokenHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.2, head_mult: float = 0.5):
        super().__init__()
        inner = max(128, int(hidden_size * head_mult))
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.LayerNorm(inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        return self.net(x).squeeze(-1)  # (B, T)


class DebertaTraceSimple(nn.Module):
    """Minimal paper-style architecture: backbone + dropout + linear heads.

    This matches the RAGBench paper: a single Linear projection per target on
    top of the backbone's last_hidden_state. Far fewer trainable parameters
    than DebertaTraceComplex, so it converges reliably in 3-5 epochs without
    overfitting random head initialization.
    """

    def __init__(self, base_model_name: str, dropout: float = 0.1):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            base_model_name, trust_remote_code=True, torch_dtype=torch.float32
        )
        hid = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rel_head = nn.Linear(hid, 1)
        self.util_head = nn.Linear(hid, 1)
        self.adh_head = nn.Linear(hid, 1)

    def forward(self, input_ids, attention_mask, **kwargs) -> Dict[str, torch.Tensor]:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hs = self.dropout(out.last_hidden_state)  # (B, T, H)
        return {
            "logits_relevance":   self.rel_head(hs).squeeze(-1),
            "logits_utilization": self.util_head(hs).squeeze(-1),
            "logits_adherence":   self.adh_head(hs).squeeze(-1),
        }


class DebertaTraceComplex(nn.Module):
    """Original complex architecture: shared transform + inner MLP heads."""

    def __init__(
        self,
        base_model_name: str,
        dropout: float = 0.2,
        shared_mult: float = 1.0,
        head_mult: float = 0.5,
        use_residual: bool = True,
    ):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            base_model_name, trust_remote_code=True, torch_dtype=torch.float32
        )
        hid = self.base.config.hidden_size
        shared_dim = max(128, int(hid * shared_mult))
        self.use_residual = use_residual
        self.shared = nn.Sequential(
            nn.Linear(hid, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.back_proj = nn.Linear(shared_dim, hid) if shared_dim != hid else nn.Identity()
        self.out_norm = nn.LayerNorm(hid)

        self.rel_head = TokenHead(hid, dropout=dropout, head_mult=head_mult)
        self.util_head = TokenHead(hid, dropout=dropout, head_mult=head_mult)
        self.adh_head = TokenHead(hid, dropout=dropout, head_mult=head_mult)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hs = out.last_hidden_state  # (B, T, H)

        x = self.shared(hs)         # (B, T, shared_dim)
        x = self.back_proj(x)       # (B, T, H)

        if self.use_residual:
            x = x + hs

        x = self.out_norm(x)        # (B, T, H)

        logits_rel = self.rel_head(x)    # (B, T)
        logits_util = self.util_head(x)  # (B, T)
        logits_adh = self.adh_head(x)    # (B, T)

        return {
            "logits_relevance": logits_rel,
            "logits_utilization": logits_util,
            "logits_adherence": logits_adh,
        }


def DebertaTrace(base_model_name: str, use_complex: bool = False, **kwargs) -> nn.Module:
    """Factory: returns simple (paper-style) or complex model.

    Args:
        base_model_name: HuggingFace model id.
        use_complex: if True → DebertaTraceComplex (shared layer + MLP heads).
                     if False (default) → DebertaTraceSimple (backbone + linear, paper-style).
        **kwargs: forwarded to the chosen class.
    """
    if use_complex:
        return DebertaTraceComplex(base_model_name, **kwargs)
    return DebertaTraceSimple(base_model_name, **kwargs)