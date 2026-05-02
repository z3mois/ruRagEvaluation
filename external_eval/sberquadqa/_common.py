"""Pure helpers for OOD scoring of trained TRACE evaluators on bearberry/sberquadqa.

Copies of the streamlit-free logic from new_streamlit.py (no streamlit imports).
Adds chunk-id tracking so token-level probs can be aggregated to chunk-level scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEPS_ROOT = REPO_ROOT / "model_utils" / "sweeps3"

TARGET_NAMES: Tuple[str, ...] = ("relevance", "utilization", "adherence")
DEFAULT_THRESHOLDS: Dict[str, float] = {"relevance": 0.5, "utilization": 0.5, "adherence": 0.5}


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "DeBERTa-v3-large (len=512)": {
        "checkpoint_dir": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "state_path": str(SWEEPS_ROOT / "MoritzLaurer__DeBERTa-v3-large-mnli-fever-anli-ling-wanli" / "len_512" / "model.pt"),
        "max_length": 512,
    },
    "ModernBERT (len=2048)": {
        "checkpoint_dir": "deepvk/RuModernBERT-base",
        "state_path": str(SWEEPS_ROOT / "deepvk__RuModernBERT-base" / "len_2048" / "model.pt"),
        "max_length": 2048,
    },
    "BAAI-bge-m3 (len=1024)": {
        "checkpoint_dir": "BAAI/bge-m3",
        "state_path": str(SWEEPS_ROOT / "BAAI__bge-m3" / "len_1024" / "model.pt"),
        "max_length": 1024,
    },
}


class DebertaTrace(nn.Module):
    """Paper-style trace head: backbone + 3 linear projections.

    Identical to the class used in new_streamlit.py — duplicated here to avoid
    importing streamlit-side code.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        hid = base_model.config.hidden_size
        self.rel_head = nn.Linear(hid, 1)
        self.util_head = nn.Linear(hid, 1)
        self.adh_head = nn.Linear(hid, 1)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        return {
            "logits_relevance": self.rel_head(hs).squeeze(-1),
            "logits_utilization": self.util_head(hs).squeeze(-1),
            "logits_adherence": self.adh_head(hs).squeeze(-1),
        }


@dataclass
class PreflightReport:
    model_key: str
    checkpoint_dir: str
    state_path: str
    state_path_exists: bool
    result_json_path: str
    result_json_exists: bool
    thresholds: Dict[str, float]
    threshold_source: str  # "result.json" | "default_0.5"
    ok: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "checkpoint_dir": self.checkpoint_dir,
            "state_path": self.state_path,
            "state_path_exists": self.state_path_exists,
            "result_json_path": self.result_json_path,
            "result_json_exists": self.result_json_exists,
            "thresholds": self.thresholds,
            "threshold_source": self.threshold_source,
            "ok": self.ok,
            "message": self.message,
        }


def preflight_check(model_key: str) -> PreflightReport:
    if model_key not in MODEL_REGISTRY:
        return PreflightReport(
            model_key=model_key,
            checkpoint_dir="", state_path="",
            state_path_exists=False,
            result_json_path="", result_json_exists=False,
            thresholds=dict(DEFAULT_THRESHOLDS),
            threshold_source="default_0.5",
            ok=False,
            message=f"Unknown model_key '{model_key}'. Known: {list(MODEL_REGISTRY)}",
        )
    entry = MODEL_REGISTRY[model_key]
    state_path = Path(entry["state_path"])
    result_path = state_path.parent / "result.json"
    state_exists = state_path.exists()
    result_exists = result_path.exists()

    thresholds = dict(DEFAULT_THRESHOLDS)
    threshold_source = "default_0.5"
    if result_exists:
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tuned = data.get("thresholds")
            if isinstance(tuned, dict) and all(t in tuned for t in TARGET_NAMES):
                thresholds = {t: float(tuned[t]) for t in TARGET_NAMES}
                threshold_source = "result.json"
        except Exception as e:  # noqa: BLE001
            threshold_source = f"default_0.5 (result.json parse failed: {e})"

    msg_parts = []
    if not state_exists:
        msg_parts.append(f"missing weights: {state_path}")
    if not result_exists:
        msg_parts.append("no result.json — using fallback thresholds 0.5")
    elif threshold_source.startswith("default_0.5"):
        msg_parts.append("result.json present but thresholds invalid — using 0.5")

    return PreflightReport(
        model_key=model_key,
        checkpoint_dir=entry["checkpoint_dir"],
        state_path=str(state_path),
        state_path_exists=state_exists,
        result_json_path=str(result_path),
        result_json_exists=result_exists,
        thresholds=thresholds,
        threshold_source=threshold_source,
        ok=state_exists,
        message="; ".join(msg_parts) if msg_parts else "ok",
    )


def load_trace_model(model_key: str, device: str):
    """Load tokenizer + model + thresholds.

    Returns: (tokenizer, model, thresholds, threshold_source, preflight)
    """
    from transformers import AutoModel, AutoTokenizer

    pre = preflight_check(model_key)
    if not pre.ok:
        raise FileNotFoundError(
            f"Preflight failed for '{model_key}': {pre.message}. "
            "Make sure the checkpoint is present at the expected path "
            "(see model_utils/sweeps3/...). This script does NOT download or train models."
        )
    entry = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(entry["checkpoint_dir"], trust_remote_code=True)
    base = AutoModel.from_pretrained(
        entry["checkpoint_dir"], trust_remote_code=True, torch_dtype=torch.float32
    )
    model = DebertaTrace(base)
    sd = torch.load(pre.state_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model, pre.thresholds, pre.threshold_source, pre


def preprocess_one_with_chunk_ids(
    example: Dict[str, Any],
    tokenizer,
    max_length: int,
    truncate: bool = False,
) -> Optional[Dict[str, Any]]:
    """Tokenise one example into tensors expected by DebertaTrace + per-token chunk_id.

    Layout (matches model_utils/data.py:build_example and new_streamlit.preprocess_one):
        [Q] [SEP] [doc tokens...] [SEP] [R]

    Returns None if the example does not fit and `truncate=False`.
    On `truncate=True`, drops chunks from the end until it fits and reports n_chunks_dropped.
    """
    question_ids = tokenizer.encode(example["question_ru"], add_special_tokens=False)
    response_ids = tokenizer.encode(example["response_ru"], add_special_tokens=False)
    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 102

    docs = example["documents_sentences_ru"]  # List[List[[key, sent]]]
    flat_chunks: List[Tuple[int, List[int]]] = []
    chunk_id = 0
    for doc in docs:
        for _key, sent in doc:
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            flat_chunks.append((chunk_id, tokens))
            chunk_id += 1
    n_chunks_total = chunk_id

    fixed_overhead = len(question_ids) + 1 + 1 + len(response_ids)  # Q + SEP + SEP + R

    def total_len(included_chunks):
        return fixed_overhead + sum(len(t) for _, t in included_chunks)

    chunks_used = list(flat_chunks)
    n_chunks_dropped = 0
    if total_len(chunks_used) > max_length:
        if not truncate:
            return None
        while chunks_used and total_len(chunks_used) > max_length:
            chunks_used.pop()
            n_chunks_dropped += 1
        if not chunks_used:
            return None

    doc_ids: List[int] = []
    chunk_id_per_doc_token: List[int] = []
    for cid, toks in chunks_used:
        doc_ids += toks
        chunk_id_per_doc_token += [cid] * len(toks)

    input_ids = question_ids + [sep_id] + doc_ids + [sep_id] + response_ids
    q_len = len(question_ids)
    d_len = len(doc_ids)
    r_len = len(response_ids)

    context_mask = [0] * (q_len + 1) + [1] * d_len + [0] + [0] * r_len
    response_mask = [0] * (q_len + 1 + d_len + 1) + [1] * r_len
    chunk_id_per_token = [-1] * (q_len + 1) + chunk_id_per_doc_token + [-1] + [-1] * r_len

    if len(input_ids) > max_length:  # safety
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
        "context_mask": torch.tensor(context_mask, dtype=torch.bool),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
        "chunk_id_per_token": torch.tensor(chunk_id_per_token, dtype=torch.long),
        "n_chunks_total": n_chunks_total,
        "n_chunks_used": len(chunks_used),
        "n_chunks_dropped": n_chunks_dropped,
    }


def run_inference_single(model: DebertaTrace, batch: Dict[str, torch.Tensor], device: str) -> Dict[str, Any]:
    input_ids = batch["input_ids"].unsqueeze(0).to(device)
    attn = batch["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn)
    return {
        "probs": {
            "relevance": torch.sigmoid(logits["logits_relevance"][0].cpu()).numpy(),
            "utilization": torch.sigmoid(logits["logits_utilization"][0].cpu()).numpy(),
            "adherence": torch.sigmoid(logits["logits_adherence"][0].cpu()).numpy(),
        },
        "masks": {
            "context": batch["context_mask"].cpu().numpy().astype(bool),
            "response": batch["response_mask"].cpu().numpy().astype(bool),
        },
    }


def aggregate_chunk_scores(
    probs: Dict[str, np.ndarray],
    chunk_id_per_token: np.ndarray,
    n_chunks_used: int,
) -> List[Dict[str, float]]:
    """For each chunk used in the input, aggregate token-level probs to mean/max."""
    rows: List[Dict[str, float]] = []
    for cid in range(n_chunks_used):
        mask = chunk_id_per_token == cid
        n_tok = int(mask.sum())
        if n_tok == 0:
            rows.append({
                "chunk_id": cid, "n_tokens": 0,
                "rel_prob_mean": float("nan"), "rel_prob_max": float("nan"),
                "util_prob_mean": float("nan"), "util_prob_max": float("nan"),
            })
            continue
        rel = probs["relevance"][mask]
        util = probs["utilization"][mask]
        rows.append({
            "chunk_id": cid,
            "n_tokens": n_tok,
            "rel_prob_mean": float(rel.mean()),
            "rel_prob_max": float(rel.max()),
            "util_prob_mean": float(util.mean()),
            "util_prob_max": float(util.max()),
        })
    return rows


def safe_model_tag(model_key: str, max_length: int) -> str:
    """Filesystem-safe tag for output dirs."""
    s = model_key.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    s = s.replace("/", "_").replace("\\", "_")
    return f"{s}__len_{max_length}"
