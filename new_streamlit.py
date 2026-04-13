import hashlib
import io
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModel, AutoTokenizer



APP_DIR = Path(__file__).resolve().parent

DATASETS = ["cuad", "delucionqa"]


REQUIRED_BATCH_FIELDS = ["question_ru", "response_ru", "documents_sentences_ru"]
OPTIONAL_BATCH_FIELDS = [
    "all_relevant_sentence_keys",
    "all_utilized_sentence_keys",
    "adherence_score",
]


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "DeBERTa-v3-large (len=512)": {
        "checkpoint_dir": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "state_path": str(
            (APP_DIR / "model_utils/sweeps3/MoritzLaurer__DeBERTa-v3-large-mnli-fever-anli-ling-wanli/len_512/model.pt").resolve()
        ),
        "max_length": 512,
        "default_thresholds": {"relevance": 0.50, "utilization": 0.50, "adherence": 0.50},
    },
    "ModernBERT (len=1024)": {
        "checkpoint_dir": "deepvk/RuModernBERT-base",
        "state_path": str(
            (APP_DIR / "model_utils/sweeps3/deepvk__RuModernBERT-base/len_1024/model.pt").resolve()
        ),
        "max_length": 1024,
        "default_thresholds": {"relevance": 0.50, "utilization": 0.50, "adherence": 0.50},
    },
    "BAAI-bge-m3 (len=1024)": {
        "checkpoint_dir": "BAAI/bge-m3",
        "state_path": str(
            (APP_DIR / "model_utils/sweeps3/BAAI__bge-m3/len_1024/model.pt").resolve()
        ),
        "max_length": 1024,
        "default_thresholds": {"relevance": 0.50, "utilization": 0.50, "adherence": 0.50},
    },
}

THRESHOLD_KEYS = ("threshold_rel", "threshold_util", "threshold_adh")
TARGET_NAMES = ("relevance", "utilization", "adherence")




class DebertaTrace(nn.Module):
    """Simple paper-style trace head: backbone + 3 linear projections.

    # ASSUMPTION: state_dict соответствует `DebertaTraceSimple` из model_utils/models.py.
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


@st.cache_resource(show_spinner=False)
def load_tokenizer_and_base(checkpoint_dir: str):
    tok = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    base = AutoModel.from_pretrained(checkpoint_dir, trust_remote_code=True, torch_dtype=torch.float32)
    return tok, base


@st.cache_resource(show_spinner=True)
def load_trace_model(checkpoint_dir: str, state_path: str, device: str):
    tokenizer, base = load_tokenizer_and_base(checkpoint_dir)
    model = DebertaTrace(base)
    sd = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

   
    tuned_thresholds = None
    result_path = Path(state_path).parent / "result.json"
    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            if "thresholds" in result_data and isinstance(result_data["thresholds"], dict):
                tuned_thresholds = result_data["thresholds"]
        except Exception:
            pass

    return tokenizer, model, {"missing": list(missing), "unexpected": list(unexpected)}, tuned_thresholds


@st.cache_data(show_spinner=False)
def load_full_ragbench() -> Dict[str, DatasetDict]:
    return {ds: load_dataset("CMCenjoyer/ragbench-ru", ds) for ds in DATASETS}


def preprocess_one(example: Dict[str, Any], tokenizer, max_length: int = 1024) -> Dict[str, torch.Tensor]:
    question_ids = tokenizer.encode(example["question_ru"], add_special_tokens=False)
    doc_ids, rel_labels, util_labels = [], [], []

    rel_keys = set(example.get("all_relevant_sentence_keys", []) or [])
    util_keys = set(example.get("all_utilized_sentence_keys", []) or [])

    for doc in example["documents_sentences_ru"]:
        for key, sent in doc:
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            doc_ids += tokens
            # Prefix match: relevant_keys like '0ac' should match document_key '0a'.
            is_rel = key in rel_keys or any(rk.startswith(key) for rk in rel_keys)
            is_util = key in util_keys or any(uk.startswith(key) for uk in util_keys)
            rel_labels += [float(is_rel)] * len(tokens)
            util_labels += [float(is_util)] * len(tokens)

    response_ids = tokenizer.encode(example["response_ru"], add_special_tokens=False)
    adh_labels = [float(example.get("adherence_score", 0.0))] * len(response_ids)

    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 102

    input_ids = question_ids + [sep_id] + doc_ids + [sep_id] + response_ids
    context_mask = [0] * (len(question_ids) + 1) + [1] * len(doc_ids) + [0] + [0] * len(response_ids)
    response_mask = [0] * (len(question_ids) + len(doc_ids) + 2) + [1] * len(response_ids)
    rel_labels = [0.0] * (len(question_ids) + 1) + rel_labels + [0.0] * (len(response_ids) + 1)
    util_labels = [0.0] * (len(question_ids) + 1) + util_labels + [0.0] * (len(response_ids) + 1)
    adh_labels = [0.0] * (len(question_ids) + len(doc_ids) + 2) + adh_labels

    if len(input_ids) > max_length:
        return {}

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
        "context_mask": torch.tensor(context_mask, dtype=torch.bool),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
        "labels_relevance": torch.tensor(rel_labels, dtype=torch.float),
        "labels_utilization": torch.tensor(util_labels, dtype=torch.float),
        "labels_adherence": torch.tensor(adh_labels, dtype=torch.float),
    }


def make_custom_example(
    question: str,
    response: str,
    docs_lines: List[str],
    relevant_line_idxs: Optional[List[int]] = None,
    utilized_line_idxs: Optional[List[int]] = None,
    adherence_score: Optional[float] = None,
) -> Dict[str, Any]:
    relevant_line_idxs = relevant_line_idxs or []
    utilized_line_idxs = utilized_line_idxs or []
    doc, rel_keys, util_keys = [], [], []
    for i, line in enumerate(docs_lines):
        key = f"s{i}"
        doc.append([key, line])
        if i in relevant_line_idxs:
            rel_keys.append(key)
        if i in utilized_line_idxs:
            util_keys.append(key)
    return {
        "question_ru": question,
        "response_ru": response,
        "documents_sentences_ru": [doc],
        "all_relevant_sentence_keys": rel_keys,
        "all_utilized_sentence_keys": util_keys,
        "adherence_score": float(adherence_score) if adherence_score is not None else 0.0,
    }


@st.cache_data(show_spinner=False)
def collect_fit_indices(
    subds: str, split: str, max_length: int, checkpoint_dir: str
) -> List[int]:
    """Indices of RAGBench examples that fit into ``max_length``. Cached per (subds,split,max_len,model)."""
    tokenizer, _ = load_tokenizer_and_base(checkpoint_dir)
    ds = st.session_state["ragbench"][subds][split]
    kept = []
    for i, ex in enumerate(ds):
        if preprocess_one(ex, tokenizer, max_length):
            kept.append(i)
    return kept


def _maybe_json(value: Any) -> Any:
    """CSV stores nested fields as strings — try parsing them."""
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def load_uploaded_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(file_bytes))
    elif name.endswith(".tsv"):
        df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
    else:  # default csv
        df = pd.read_csv(io.BytesIO(file_bytes))

    missing = [f for f in REQUIRED_BATCH_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"В файле не хватает обязательных полей: {missing}")

    # parse JSON-encoded fields
    for col in ("documents_sentences_ru", "all_relevant_sentence_keys", "all_utilized_sentence_keys"):
        if col in df.columns:
            df[col] = df[col].apply(_maybe_json)
    return df


def dataframe_row_to_example(row: pd.Series) -> Dict[str, Any]:
    return {
        "question_ru": row["question_ru"],
        "response_ru": row["response_ru"],
        "documents_sentences_ru": row["documents_sentences_ru"],
        "all_relevant_sentence_keys": row.get("all_relevant_sentence_keys", []) or [],
        "all_utilized_sentence_keys": row.get("all_utilized_sentence_keys", []) or [],
        "adherence_score": float(row.get("adherence_score", 0.0) or 0.0),
    }

def _bump_inference_counter():
    st.session_state["inference_calls"] = st.session_state.get("inference_calls", 0) + 1


def run_inference_single(model: DebertaTrace, batch: Dict[str, torch.Tensor], device: str) -> Dict[str, Any]:
    input_ids = batch["input_ids"].unsqueeze(0).to(device)
    attn = batch["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn)
    _bump_inference_counter()
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
        "labels": {
            "relevance": batch["labels_relevance"].cpu().numpy(),
            "utilization": batch["labels_utilization"].cpu().numpy(),
            "adherence": batch["labels_adherence"].cpu().numpy(),
        },
    }

def build_display_tokens_from_ids(input_ids: List[int], tokenizer) -> List[str]:
    """
    Human-readable per-token display.
    Works better for byte-level / ModernBERT-like tokenizers than convert_ids_to_tokens().
    Keeps 1 display item per input id.
    """
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])

    display_tokens: List[str] = []
    for tid, raw_tok in zip(input_ids, raw_tokens):
        # Keep special tokens explicit
        if raw_tok in special_tokens:
            display_tokens.append(raw_tok)
            continue

        piece = ""
        try:
            piece = tokenizer.decode(
                [int(tid)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            piece = ""

        if piece is None or piece == "":
            piece = raw_tok
            if piece.startswith("##"):
                piece = piece[2:]
            if piece.startswith("▁") or piece.startswith("Ġ"):
                piece = piece[1:]
        piece = (
            piece.replace("\n", "\\n")
                 .replace("\t", "\\t")
                 .replace("\r", "\\r")
        )

        if piece == " ":
            piece = "␠"
        elif piece == "":
            piece = "∅"

        display_tokens.append(piece)

    return display_tokens
def run_inference_batch(
    model: DebertaTrace,
    tokenizer,
    examples: List[Dict[str, Any]],
    max_length: int,
    device: str,
) -> List[Optional[Dict[str, Any]]]:
    results: List[Optional[Dict[str, Any]]] = []
    progress = st.progress(0.0, text="Inference…")
    n = max(len(examples), 1)
    for i, ex in enumerate(examples):
        batch = preprocess_one(ex, tokenizer, max_length)
        if not batch:
            results.append(None)
        else:
            res = run_inference_single(model, batch, device)
            input_ids_list = batch["input_ids"].tolist()
            res["input_ids"] = input_ids_list
            res["tokens"] = tokenizer.convert_ids_to_tokens(input_ids_list)
            res["display_tokens"] = build_display_tokens_from_ids(input_ids_list, tokenizer)
            results.append(res)
        progress.progress((i + 1) / n, text=f"Inference… {i+1}/{n}")
    progress.empty()
    return results


def apply_thresholds(
    probs: Dict[str, np.ndarray], thresholds: Dict[str, float]
) -> Dict[str, np.ndarray]:
    return {t: probs[t] > thresholds[t] for t in TARGET_NAMES}


def compute_metrics_from_preds(
    preds: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]
) -> Dict[str, float]:
    ctx, resp = masks["context"], masks["response"]

    def rate(p: np.ndarray, m: np.ndarray) -> float:
        denom = max(int(m.sum()), 1)
        return float((p & m).sum()) / denom

    rel_rate = rate(preds["relevance"], ctx)
    util_rate = rate(preds["utilization"], ctx)
    adh_rate = rate(preds["adherence"], resp)

    num_ru = int((preds["relevance"] & preds["utilization"] & ctx).sum())
    den_r = max(int(preds["relevance"].sum()), 1)
    completeness = num_ru / den_r

    return {
        "relevance_rate": rel_rate,
        "utilization_rate": util_rate,
        "adherence_rate": adh_rate,
        "completeness": completeness,
    }


def aggregate_batch_metrics(per_example: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_example:
        return {}
    keys = per_example[0].keys()
    return {k: float(np.mean([m[k] for m in per_example])) for k in keys}


def compute_true_metrics(
    labels: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]
) -> Optional[Dict[str, float]]:
    """Compute GT-based rates from true labels. Returns None if no positive labels exist."""
    ctx, resp = masks["context"], masks["response"]
    true_rel = (labels["relevance"] > 0.5) & ctx
    true_util = (labels["utilization"] > 0.5) & ctx
    true_adh = (labels["adherence"] > 0.5) & resp

    if not true_rel.any() and not true_util.any() and not true_adh.any():
        return None

    def rate(t: np.ndarray, m: np.ndarray) -> float:
        return float(t.sum()) / max(int(m.sum()), 1)

    rel_rate = rate(true_rel, ctx)
    util_rate = rate(true_util, ctx)
    adh_rate = rate(true_adh, resp)

    den_r = max(int(true_rel.sum()), 1)
    completeness = float((true_rel & true_util).sum()) / den_r

    return {
        "relevance_rate": rel_rate,
        "utilization_rate": util_rate,
        "adherence_rate": adh_rate,
        "completeness": completeness,
    }


def render_single_metrics(
    metrics: Dict[str, float],
    true_metrics: Optional[Dict[str, float]] = None,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevance (ctx)", f"{metrics['relevance_rate']:.3f}")
    c2.metric("Utilization (ctx)", f"{metrics['utilization_rate']:.3f}")
    c3.metric("Adherence (resp)", f"{metrics['adherence_rate']:.3f}")
    c4.metric("Completeness", f"{metrics['completeness']:.3f}")

    if true_metrics:
        st.caption("Истинные значения (GT):")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("True Relevance", f"{true_metrics['relevance_rate']:.3f}")
        t2.metric("True Utilization", f"{true_metrics['utilization_rate']:.3f}")
        t3.metric("True Adherence", f"{true_metrics['adherence_rate']:.3f}")
        t4.metric("True Completeness", f"{true_metrics['completeness']:.3f}")
        
def build_token_display_columns(
    raw_tokens: List[str],
    tokenizer,
) -> pd.DataFrame:
    """
    Build human-readable token display without breaking 1-row-per-token alignment.

    Returns columns:
      - raw_token: original tokenizer token
      - token: cleaned token for UI
      - word_start: 1 if token starts a new word (SentencePiece/BPE marker)
      - is_special: 1 if tokenizer special token
    """
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])

    rows = []
    for tok in raw_tokens:
        word_start = 0
        is_special = int(tok in special_tokens)

        clean = tok

        if not is_special:
            # SentencePiece / byte-level BPE
            if clean.startswith("▁") or clean.startswith("Ġ"):
                word_start = 1
                clean = clean[1:]

            # WordPiece
            if clean.startswith("##"):
                clean = clean[2:]
            if clean == "":
                clean = "∅"

        rows.append(
            {
                "raw_token": tok,
                "token": clean,
                "word_start": word_start,
                "is_special": is_special,
            }
        )

    return pd.DataFrame(rows)
def build_token_display_columns(
    raw_tokens: List[str],
    tokenizer,
) -> pd.DataFrame:
    """
    Build human-readable token display without breaking 1-row-per-token alignment.

    Returns columns:
      - raw_token: original tokenizer token
      - token: cleaned token for UI
      - word_start: 1 if token starts a new word (SentencePiece/BPE marker)
      - is_special: 1 if tokenizer special token
    """
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])

    rows = []
    for tok in raw_tokens:
        word_start = 0
        is_special = int(tok in special_tokens)

        clean = tok

        if not is_special:
            if clean.startswith("▁") or clean.startswith("Ġ"):
                word_start = 1
                clean = clean[1:]
            if clean.startswith("##"):
                clean = clean[2:]
            if clean == "":
                clean = "∅"

        rows.append(
            {
                "raw_token": tok,
                "token": clean,
                "word_start": word_start,
                "is_special": is_special,
            }
        )

    return pd.DataFrame(rows)


def render_token_table(
    tokens: List[str],
    probs: Dict[str, np.ndarray],
    preds: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    *,
    display_tokens: Optional[List[str]] = None,
    labels: Optional[Dict[str, np.ndarray]] = None,
    prob_round: int = 3,
    only_ctx: bool = False,
    only_resp: bool = False) -> None:
    token_vis = display_tokens if display_tokens is not None else tokens

    data: Dict[str, Any] = {
        "token": token_vis,
        "raw_token": tokens,
        "is_ctx": masks["context"].astype(int),
        "is_resp": masks["response"].astype(int),
        "rel_prob": probs["relevance"],
        "util_prob": probs["utilization"],
        "adh_prob": probs["adherence"],
        "rel_pred": preds["relevance"].astype(int),
        "util_pred": preds["utilization"].astype(int),
        "adh_pred": preds["adherence"].astype(int),
    }
    if labels is not None:
        data["true_rel"] = ((labels["relevance"] > 0.5) & masks["context"]).astype(int)
        data["true_util"] = ((labels["utilization"] > 0.5) & masks["context"]).astype(int)
        data["true_adh"] = ((labels["adherence"] > 0.5) & masks["response"]).astype(int)

    df = pd.DataFrame(data)

    if only_ctx:
        df = df[df["is_ctx"] == 1]
    if only_resp:
        df = df[df["is_resp"] == 1]

    for c in ("rel_prob", "util_prob", "adh_prob"):
        df[c] = df[c].round(prob_round)

    legend_parts = [
        "<span style='background:#fde2c4;padding:2px 6px;border-radius:4px'>rel_prob</span>",
        "<span style='background:#c4e2fd;padding:2px 6px;border-radius:4px'>util_prob</span>",
        "<span style='background:#d2fdc4;padding:2px 6px;border-radius:4px'>adh_prob</span>",
        "<span style='background:#eeeeee;padding:2px 6px;border-radius:4px'>word_start=1</span>",
        "<span style='background:#eeeeee;padding:2px 6px;border-radius:4px'>is_special=1</span>",
    ]
    if labels is not None:
        legend_parts.append(
            "<span style='background:#f5c6c6;padding:2px 6px;border-radius:4px'>true (GT)</span>"
        )
    st.markdown("**Легенда:** " + " ".join(legend_parts), unsafe_allow_html=True)
    preferred_cols = [
        "token", "word_start", "is_special",
        "is_ctx", "is_resp",
        "rel_prob", "util_prob", "adh_prob",
        "rel_pred", "util_pred", "adh_pred",
        "true_rel", "true_util", "true_adh",
        "raw_token",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    df = df[cols]

    try:
        styled = (
            df.style
            .background_gradient(subset=["rel_prob"], cmap="Oranges", vmin=0, vmax=1)
            .background_gradient(subset=["util_prob"], cmap="Blues", vmin=0, vmax=1)
            .background_gradient(subset=["adh_prob"], cmap="Greens", vmin=0, vmax=1)
        )
        if labels is not None:
            styled = (
                styled
                .background_gradient(subset=["true_rel"], cmap="Reds", vmin=0, vmax=1)
                .background_gradient(subset=["true_util"], cmap="Reds", vmin=0, vmax=1)
                .background_gradient(subset=["true_adh"], cmap="Reds", vmin=0, vmax=1)
            )
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)

def render_batch_results(df: pd.DataFrame, agg: Dict[str, float]) -> None:
    st.subheader("Агрегированные метрики (среднее по датасету)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevance",   f"{agg.get('relevance_rate', 0):.3f}")
    c2.metric("Utilization", f"{agg.get('utilization_rate', 0):.3f}")
    c3.metric("Adherence",   f"{agg.get('adherence_rate', 0):.3f}")
    c4.metric("Completeness",f"{agg.get('completeness', 0):.3f}")

    st.subheader("Результаты по примерам")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "📥 Скачать CSV с метриками",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="scored.csv",
        mime="text/csv",
    )


st.set_page_config(page_title="Trace Inference", layout="wide")
st.title("🧪 Trace Inference")


with st.spinner("Загрузка RAGBench-ru (один раз)..."):
    st.session_state["ragbench"] = load_full_ragbench()
ragbench = st.session_state["ragbench"]


with st.sidebar:
    st.header("⚙️ Модель")
    model_key = st.selectbox("Модель", options=list(MODEL_REGISTRY.keys()), index=0)
    entry = MODEL_REGISTRY[model_key]
    max_length = int(entry["max_length"])
    st.caption(f"max_length: {max_length}")

    prev_key = st.session_state.get("active_model_key")
    if prev_key != model_key:
        for k in THRESHOLD_KEYS:
            st.session_state.pop(k, None)
        st.session_state.pop("single", None)
        st.session_state.pop("batch", None)
        st.session_state["active_model_key"] = model_key

    tuned = st.session_state.get(f"tuned_thresholds_{model_key}")
    defaults = tuned if tuned else entry["default_thresholds"]
    source_label = "(из артефакта модели)" if tuned else "(дефолтные)"
    st.markdown(
        f"**Рекомендуемые пороги {source_label}:** "
        f"rel={defaults['relevance']:.2f}, "
        f"util={defaults['utilization']:.2f}, "
        f"adh={defaults['adherence']:.2f}"
    )
    threshold_rel = st.slider("Порог relevance", 0.05, 0.95, defaults["relevance"], 0.01, key="threshold_rel")
    threshold_util = st.slider("Порог utilization", 0.05, 0.95, defaults["utilization"], 0.01, key="threshold_util")
    threshold_adh = st.slider("Порог adherence", 0.05, 0.95, defaults["adherence"], 0.01, key="threshold_adh")
    thresholds = {"relevance": threshold_rel, "utilization": threshold_util, "adherence": threshold_adh}

    auto_device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    device = st.selectbox(
        "Девайс", options=["cpu", "cuda", "mps"],
        index=["cpu", "cuda", "mps"].index(auto_device), key="device",
    )

    st.divider()
    st.header("📌 Режим")
    mode = st.radio("Источник входа", options=["RAGBench example", "Custom input", "Batch file"], index=0)
    if mode == "RAGBench example":
        subds = st.selectbox("Поддатасет", options=DATASETS, index=0)
        split = st.selectbox("Сплит", options=["train", "validation", "test"], index=0)

    st.divider()
    st.caption(f"inference calls: {st.session_state.get('inference_calls', 0)}")

if not os.path.exists(entry["state_path"]):
    st.error(
        f"Файл весов не найден: `{entry['state_path']}`\n\n"
        "Обучи модель через `run_experiments.py` или укажи правильный путь в `MODEL_REGISTRY`."
    )
    st.stop()

try:
    with st.spinner("Загружаю модель (cache_resource)..."):
        tokenizer, model, sd_info, tuned_thresholds = load_trace_model(
            entry["checkpoint_dir"], entry["state_path"], device
        )
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
    st.stop()

if tuned_thresholds is not None:
    prev_tuned = st.session_state.get(f"tuned_thresholds_{model_key}")
    if prev_tuned != tuned_thresholds:
        st.session_state[f"tuned_thresholds_{model_key}"] = tuned_thresholds
        for k in THRESHOLD_KEYS:
            st.session_state.pop(k, None)
        st.rerun()

if sd_info["missing"] or sd_info["unexpected"]:
    with st.expander("⚠️ Загрузка весов: missing / unexpected"):
        st.write({"missing": sd_info["missing"], "unexpected": sd_info["unexpected"]})

if mode == "RAGBench example":
    if subds not in ragbench or split not in ragbench[subds]:
        st.error("Некорректный поддатасет / сплит.")
        st.stop()

    current_split: Dataset = ragbench[subds][split]
    with st.spinner("Фильтрую элементы по длине (кэш по subds/split/len/model)..."):
        fit_indices = collect_fit_indices(subds, split, max_length, entry["checkpoint_dir"])

    if not fit_indices:
        st.error("Ни один элемент не подходит по длине. Выбери другой сплит/модель.")
        st.stop()

    left, right = st.columns([1, 3], vertical_alignment="top")
    with left:
        st.subheader("Выбор элемента")
        st.caption(f"`{subds}` / `{split}` — подходит {len(fit_indices)}")
        pos = st.number_input("Позиция", min_value=0, max_value=len(fit_indices) - 1, value=0, step=1)
        chosen_idx = fit_indices[int(pos)]
        src_ex = current_split[int(chosen_idx)]
        with st.expander("Сырой пример (усечённо)"):
            st.json({
                "question": src_ex.get("question_ru", "")[:500],
                "response": src_ex.get("response_ru", "")[:500],
                "adherence_score": src_ex.get("adherence_score", None),
                "relevant_keys_cnt": len(src_ex.get("all_relevant_sentence_keys", [])),
                "utilized_keys_cnt": len(src_ex.get("all_utilized_sentence_keys", [])),
            })
        run = st.button("🚀 Запустить инференс", key="run_dataset", type="primary")

    example_id = ("ragbench", subds, split, int(pos))
    single_key = (model_key, example_id, max_length)

    if run:
        batch = preprocess_one(src_ex, tokenizer, max_length)
        if not batch:
            with right:
                st.error("Не влазит в max_length.")
            st.stop()
        res = run_inference_single(model, batch, device)
        input_ids_list = batch["input_ids"].tolist()
        res["input_ids"] = input_ids_list
        res["tokens"] = tokenizer.convert_ids_to_tokens(input_ids_list)          # debug/raw
        res["display_tokens"] = build_display_tokens_from_ids(input_ids_list, tokenizer)  # UI
        res["key"] = single_key
        st.session_state["single"] = res

    with right:
        st.subheader("Результаты")
        single = st.session_state.get("single")
        if not single or single.get("key", (None,))[0] != model_key:
            st.info("Нажми «Запустить инференс». Изменение порогов не вызывает пересчёт модели.")
            st.stop()

        preds = apply_thresholds(single["probs"], thresholds)
        metrics = compute_metrics_from_preds(preds, single["masks"])
        true_metrics = compute_true_metrics(single["labels"], single["masks"])
        render_single_metrics(metrics, true_metrics=true_metrics)

        st.divider()
        cprob, cctx, cresp = st.columns([1, 1, 1])
        prob_round = cprob.slider("Округление вероятностей", 2, 4, 3)
        only_ctx = cctx.checkbox("Только контекстные токены", value=False)
        only_resp = cresp.checkbox("Только токены ответа", value=False)
        render_token_table(
            single["tokens"], single["probs"], preds, single["masks"],
            display_tokens=single.get("display_tokens"),
            labels=single.get("labels"),
            prob_round=prob_round, only_ctx=only_ctx, only_resp=only_resp,
        )

elif mode == "Custom input":
    left, right = st.columns([1, 3], vertical_alignment="top")
    with left:
        st.subheader("Custom input")
        q = st.text_area("Вопрос", height=120, key="custom_q")
        r = st.text_area("Ответ", height=120, key="custom_r")
        docs = st.text_area("Документы (1 строка = 1 предложение)", height=220, key="custom_docs")
        rel_idx = st.text_input("Relevance line idx (0,2,5)", value="", key="custom_rel")
        util_idx = st.text_input("Utilization line idx (1,2)", value="", key="custom_util")
        adh_score = st.number_input(
            "Ответ корректен? (adherence GT)",
            min_value=0.0, max_value=1.0, value=1.0, step=1.0,
            help="Ground truth: 1.0 = ответ полностью корректен, 0.0 = нет. "
                 "Используется для отображения истинных меток в таблице.",
            key="custom_adh",
        )

        def _parse_idxs(s: str) -> List[int]:
            return [int(x.strip()) for x in s.split(",") if x.strip()]

        run = st.button("🚀 Запустить инференс", key="run_custom", type="primary")

    if run:
        docs_lines = [ln.strip() for ln in docs.splitlines() if ln.strip()]
        if not q.strip() or not r.strip() or not docs_lines:
            with right:
                st.error("Нужны вопрос, ответ и хотя бы одна строка контекста.")
            st.stop()
        custom_ex = make_custom_example(
            question=q.strip(), response=r.strip(), docs_lines=docs_lines,
            relevant_line_idxs=_parse_idxs(rel_idx),
            utilized_line_idxs=_parse_idxs(util_idx),
            adherence_score=float(adh_score),
        )
        ex_hash = hashlib.md5(json.dumps(custom_ex, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        single_key = (model_key, ("custom", ex_hash), max_length)
        batch = preprocess_one(custom_ex, tokenizer, max_length)
        if not batch:
            with right:
                st.error("Не влазит в max_length.")
            st.stop()
        res = run_inference_single(model, batch, device)
        input_ids_list = batch["input_ids"].tolist()
        res["input_ids"] = input_ids_list
        res["tokens"] = tokenizer.convert_ids_to_tokens(input_ids_list)          # debug/raw
        res["display_tokens"] = build_display_tokens_from_ids(input_ids_list, tokenizer)  # UI
        res["key"] = single_key
        st.session_state["single"] = res

    with right:
        st.subheader("Результаты")
        single = st.session_state.get("single")
        # Show results if inference was done for current model (any custom example).
        if not single or single.get("key", (None,))[0] != model_key:
            st.info("Заполни поля и нажми «Запустить инференс».")
            st.stop()

        preds = apply_thresholds(single["probs"], thresholds)
        metrics = compute_metrics_from_preds(preds, single["masks"])
        true_metrics = compute_true_metrics(single["labels"], single["masks"])
        render_single_metrics(metrics, true_metrics=true_metrics)

        st.divider()
        cprob, cctx, cresp = st.columns([1, 1, 1])
        prob_round = cprob.slider("Округление вероятностей", 2, 4, 3, key="cust_round")
        only_ctx = cctx.checkbox("Только контекстные токены", value=False, key="cust_only_ctx")
        only_resp = cresp.checkbox("Только токены ответа", value=False, key="cust_only_resp")
        render_token_table(
            single["tokens"], single["probs"], preds, single["masks"],
            display_tokens=single.get("display_tokens"),
            labels=single.get("labels"),
            prob_round=prob_round, only_ctx=only_ctx, only_resp=only_resp,
        )
else:
    st.subheader("Batch scoring из файла")
    st.info(f"Device: **{device}**. Required fields: {REQUIRED_BATCH_FIELDS}")

    uploaded = st.file_uploader("Загрузи CSV / TSV / Parquet", type=["csv", "tsv", "parquet"])
    if uploaded is None:
        st.stop()

    file_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    try:
        df_raw = load_uploaded_dataset(file_bytes, uploaded.name)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Не смог прочитать файл: {e}")
        st.stop()

    st.caption(f"Загружено строк: **{len(df_raw)}**")
    if len(df_raw) > 2000:
        st.warning("Файл большой (>2000). Inference и хранение probs займёт время/память.")

    batch_key = (model_key, file_hash, max_length)
    cached = st.session_state.get("batch")

    run_batch = st.button("🚀 Запустить batch inference", type="primary")
    if run_batch or (cached and cached.get("key") == batch_key):
        if not cached or cached.get("key") != batch_key:
            examples = [dataframe_row_to_example(row) for _, row in df_raw.iterrows()]
            results = run_inference_batch(model, tokenizer, examples, max_length, device)
            st.session_state["batch"] = {"key": batch_key, "results": results, "df": df_raw}
            cached = st.session_state["batch"]

        results = cached["results"]
        df_out = cached["df"].copy()

        per_example_metrics: List[Dict[str, float]] = []
        cols = {"relevance_rate": [], "utilization_rate": [], "adherence_rate": [], "completeness": []}
        for res in results:
            if res is None:
                for k in cols:
                    cols[k].append(np.nan)
                continue
            preds = apply_thresholds(res["probs"], thresholds)
            m = compute_metrics_from_preds(preds, res["masks"])
            per_example_metrics.append(m)
            for k in cols:
                cols[k].append(m[k])

        for k, v in cols.items():
            df_out[k] = v

        agg = aggregate_batch_metrics(per_example_metrics)
        render_batch_results(df_out, agg)
    else:
        st.info("Нажми «Запустить batch inference». При повторе с теми же файлом+моделью inference не запустится — пересчёт только метрик.")

st.caption(
    "Кэширование: датасет — cache_data, модели — cache_resource, logits — session_state. "
    "Изменение порогов НЕ запускает повторный inference."
)
