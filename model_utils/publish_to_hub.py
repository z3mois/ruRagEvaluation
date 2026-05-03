"""Package a trained TRACE checkpoint from `model_utils/sweeps3/...` and publish
it to the HuggingFace Hub as a stand-alone model release.

What this script does:
    1. Preflight: reads `model.pt` + `result.json` next to it, verifies the
       weights match the chosen architecture (simple/complex).
    2. Builds an HF release folder containing:
         - model.safetensors (full state_dict, secure format)
         - tokenizer files (copied from the training dir or downloaded fresh)
         - trace_config.json (architecture, max_length, thresholds, metrics)
         - thresholds.json
         - README.md (model card)
         - examples/inference_minimal.py
         - examples/inference_with_chunks.py
    3. Uploads the folder via huggingface_hub.HfApi (or stays local with
       --dry_run).

This script does NOT retrain or modify the checkpoint. It does NOT change any
existing scoring / inference code in the repo.

Usage:
    python model_utils/publish_to_hub.py \
        --model_key "ModernBERT (len=2048)" \
        --repo_id <owner>/ru-trace-modernbert-2048 \
        --dry_run

    python model_utils/publish_to_hub.py \
        --model_key "ModernBERT (len=2048)" \
        --repo_id <owner>/ru-trace-modernbert-2048 \
        --private
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---- Repo-local imports (model_utils/ on sys.path or run with -m) -----------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# We reuse the already-existing MODEL_REGISTRY from external_eval to avoid
# duplicating checkpoint paths and max_length defaults.
sys.path.insert(0, str(REPO_ROOT))
from external_eval.sberquadqa._common import (  # noqa: E402
    MODEL_REGISTRY,
    preflight_check,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _load_state_dict(state_path: Path) -> Dict[str, "torch.Tensor"]:
    import torch
    try:
        sd = torch.load(str(state_path), map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(str(state_path), map_location="cpu", weights_only=False)
    # Make sure tensors are detached & on CPU before serialising.
    return {k: v.detach().cpu().contiguous() for k, v in sd.items()}


def _detect_architecture(state_dict: Dict[str, "torch.Tensor"]) -> str:
    """Return 'simple' or 'complex' based on the keys present in the state_dict.

    Simple   ⇒ rel_head.weight (Linear)
    Complex  ⇒ shared.0.weight + rel_head.net.0.weight (MLP)
    """
    has_shared = any(k.startswith("shared.") for k in state_dict)
    has_mlp_head = any(k.startswith("rel_head.net.") for k in state_dict)
    if has_shared or has_mlp_head:
        return "complex"
    if any(k == "rel_head.weight" for k in state_dict):
        return "simple"
    raise ValueError(
        "Unrecognised state_dict layout: cannot find rel_head.weight nor "
        "shared.* / rel_head.net.* keys. Is this a TRACE checkpoint?"
    )


def _build_model(backbone: str, architecture: str) -> "torch.nn.Module":
    """Instantiate DebertaTraceSimple / DebertaTraceComplex via the factory in
    model_utils/models.py. We do this lazily so importing this script never
    touches the Internet."""
    from models import DebertaTrace  # noqa: WPS433
    return DebertaTrace(backbone, use_complex=(architecture == "complex"))


def _safetensors_save(state_dict: Dict[str, "torch.Tensor"], out_path: Path):
    from safetensors.torch import save_file
    save_file(state_dict, str(out_path))


def _copy_tokenizer(state_dir: Path, backbone: str, dst_dir: Path) -> str:
    """Copy tokenizer files from the training dir if they exist, otherwise
    download them fresh from the backbone HF id. Returns 'copied' or
    'downloaded' for logging."""
    tok_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ]
    copied_any = False
    for name in tok_files:
        src = state_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied_any = True
    if copied_any:
        _normalise_tokenizer_config(dst_dir)
        return "copied"

    # Fallback: pull tokenizer fresh from the backbone.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    tok.save_pretrained(str(dst_dir))
    _normalise_tokenizer_config(dst_dir)
    return "downloaded"


# Some upstream backbones ship tokenizer_config.json with a `tokenizer_class`
# value that does not exist in standard transformers (e.g. RuModernBERT-base
# uses "TokenizersBackend"). AutoTokenizer.from_pretrained then refuses to
# load. Replace those values with `PreTrainedTokenizerFast` when tokenizer.json
# is present, since tokenizers backed by a single tokenizer.json file always
# load via the fast wrapper.
_BAD_TOKENIZER_CLASSES = {"TokenizersBackend"}


def _normalise_tokenizer_config(dst_dir: Path):
    cfg_path = dst_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[publish] warning: cannot read {cfg_path}: {e}")
        return
    cur = cfg.get("tokenizer_class")
    if cur in _BAD_TOKENIZER_CLASSES and (dst_dir / "tokenizer.json").exists():
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[publish] normalised tokenizer_class: {cur} -> PreTrainedTokenizerFast")


def _copy_backbone_config(backbone: str, dst_dir: Path):
    """Save the backbone config.json into the release folder so HF can locate
    architecture metadata even without an Internet round-trip."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(backbone, trust_remote_code=True)
    cfg.save_pretrained(str(dst_dir))


# ----------------------------------------------------------------------------
# Examples (inlined as text — we want them stand-alone, no repo dependency)
# ----------------------------------------------------------------------------

EXAMPLE_INFERENCE_MINIMAL = '''\
"""Minimal inference example for a stand-alone TRACE checkpoint published
on HuggingFace Hub. Does NOT depend on the ruRagEvaluation repo.

Usage:
    pip install torch transformers safetensors huggingface_hub
    python inference_minimal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


REPO_ID = "{repo_id}"


class DebertaTraceSimple(nn.Module):
    """Same minimal architecture as in the original training repo."""

    def __init__(self, backbone: str, dropout: float = 0.1):
        super().__init__()
        # attn_implementation="eager" disables flash-attention, which is
        # required for CPU inference and harmless on GPU.
        self.base = AutoModel.from_pretrained(
            backbone, trust_remote_code=True, attn_implementation="eager"
        )
        hid = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rel_head = nn.Linear(hid, 1)
        self.util_head = nn.Linear(hid, 1)
        self.adh_head = nn.Linear(hid, 1)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.dropout(out.last_hidden_state)
        return {{
            "logits_relevance":   self.rel_head(hs).squeeze(-1),
            "logits_utilization": self.util_head(hs).squeeze(-1),
            "logits_adherence":   self.adh_head(hs).squeeze(-1),
        }}


def main():
    snapshot_dir = Path(snapshot_download(REPO_ID))
    cfg = json.loads((snapshot_dir / "trace_config.json").read_text(encoding="utf-8"))
    backbone = cfg["backbone"]
    thresholds = cfg["thresholds"]
    print(f"backbone={{backbone}}, thresholds={{thresholds}}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir), trust_remote_code=True)
    model = DebertaTraceSimple(backbone)
    state = load_file(str(snapshot_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()

    # Toy example — replace with real RAG sample.
    question = "Кто впервые показал синематограф в Париже?"
    docs = "22 марта 1895 года в Париже братьями Люмьер был впервые продемонстрирован их синематограф."
    answer = "братьями Люмьер"

    # Build the input manually so we know exactly where context and response
    # tokens live → required for proper utilization / adherence aggregation.
    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    d_ids = tokenizer.encode(docs, add_special_tokens=False)
    r_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_ids = q_ids + [sep_id] + d_ids + [sep_id] + r_ids
    if len(input_ids) > cfg["max_length"]:
        raise ValueError(f"input too long: {{len(input_ids)}} > {{cfg['max_length']}}")

    ii = torch.tensor(input_ids).unsqueeze(0)
    am = torch.ones_like(ii)
    out = model(input_ids=ii, attention_mask=am)
    rel = torch.sigmoid(out["logits_relevance"][0]).numpy()
    util = torch.sigmoid(out["logits_utilization"][0]).numpy()
    adh = torch.sigmoid(out["logits_adherence"][0]).numpy()

    ctx_start = len(q_ids) + 1
    ctx_end = ctx_start + len(d_ids)
    resp_start = ctx_end + 1
    resp_end = resp_start + len(r_ids)

    mean_rel_ctx = float(rel[ctx_start:ctx_end].mean()) if len(d_ids) else float("nan")
    mean_util_ctx = float(util[ctx_start:ctx_end].mean()) if len(d_ids) else float("nan")
    mean_adh_resp = float(adh[resp_start:resp_end].mean()) if len(r_ids) else float("nan")

    thr_rel = thresholds["relevance"]
    thr_util = thresholds["utilization"]
    thr_adh = thresholds["adherence"]

    print()
    print("=== TRACe scores ===")
    print(f"relevance   (mean over context tokens):  {{mean_rel_ctx:.3f}}  "
          f"thr={{thr_rel:.2f}} -> context_relevant={{mean_rel_ctx > thr_rel}}")
    print(f"utilization (mean over context tokens):  {{mean_util_ctx:.3f}}  "
          f"thr={{thr_util:.2f}} -> context_used={{mean_util_ctx > thr_util}}")
    print(f"adherence   (mean over response tokens): {{mean_adh_resp:.3f}}  "
          f"thr={{thr_adh:.2f}} -> answer_grounded={{mean_adh_resp > thr_adh}}")


if __name__ == "__main__":
    main()
'''


EXAMPLE_INFERENCE_WITH_CHUNKS = '''\
"""Chunk-level inference example: aggregate token-level relevance probs back
to chunks and compare against thresholds from the model card.

Stand-alone: does NOT depend on the ruRagEvaluation repo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

REPO_ID = "{repo_id}"


class DebertaTraceSimple(nn.Module):
    def __init__(self, backbone: str, dropout: float = 0.1):
        super().__init__()
        # attn_implementation="eager" disables flash-attention, which is
        # required for CPU inference and harmless on GPU.
        self.base = AutoModel.from_pretrained(
            backbone, trust_remote_code=True, attn_implementation="eager"
        )
        hid = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rel_head = nn.Linear(hid, 1)
        self.util_head = nn.Linear(hid, 1)
        self.adh_head = nn.Linear(hid, 1)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.dropout(out.last_hidden_state)
        return {{
            "logits_relevance":   self.rel_head(hs).squeeze(-1),
            "logits_utilization": self.util_head(hs).squeeze(-1),
            "logits_adherence":   self.adh_head(hs).squeeze(-1),
        }}


def score_chunks(question: str, chunks: List[str], answer: str,
                 model: nn.Module, tokenizer, max_length: int):
    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    r_ids = tokenizer.encode(answer, add_special_tokens=False)

    doc_ids: List[int] = []
    chunk_id_per_token: List[int] = []
    for cid, chunk in enumerate(chunks):
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        doc_ids += ids
        chunk_id_per_token += [cid] * len(ids)

    input_ids = q_ids + [sep_id] + doc_ids + [sep_id] + r_ids
    chunk_ids = ([-1] * (len(q_ids) + 1)
                 + chunk_id_per_token
                 + [-1] + [-1] * len(r_ids))
    if len(input_ids) > max_length:
        raise ValueError(f"Input too long: {{len(input_ids)}} > {{max_length}}")

    ii = torch.tensor(input_ids).unsqueeze(0)
    am = torch.ones_like(ii)
    out = model(input_ids=ii, attention_mask=am)
    rel = torch.sigmoid(out["logits_relevance"][0]).numpy()
    util = torch.sigmoid(out["logits_utilization"][0]).numpy()
    adh = torch.sigmoid(out["logits_adherence"][0]).numpy()

    chunk_ids_arr = torch.tensor(chunk_ids).numpy()
    chunk_rows = []
    for cid, chunk_text in enumerate(chunks):
        mask = chunk_ids_arr == cid
        n = int(mask.sum())
        if n == 0:
            chunk_rows.append({{
                "chunk_id": cid,
                "n_tokens": 0,
                "rel_prob_mean": float("nan"), "rel_prob_max": float("nan"),
                "util_prob_mean": float("nan"), "util_prob_max": float("nan"),
            }})
            continue
        chunk_rows.append({{
            "chunk_id": cid,
            "n_tokens": n,
            "rel_prob_mean":  float(rel[mask].mean()),
            "rel_prob_max":   float(rel[mask].max()),
            "util_prob_mean": float(util[mask].mean()),
            "util_prob_max":  float(util[mask].max()),
        }})

    # Adherence is a per-response quantity (it asks "are response tokens grounded
    # in the context?"), so it is aggregated over response tokens, not chunks.
    resp_start = len(q_ids) + 1 + len(doc_ids) + 1
    resp_end = resp_start + len(r_ids)
    mean_adh_resp = float(adh[resp_start:resp_end].mean()) if len(r_ids) else float("nan")

    return {{
        "chunks": chunk_rows,
        "mean_adherence_over_response": mean_adh_resp,
    }}


def main():
    snapshot_dir = Path(snapshot_download(REPO_ID))
    cfg = json.loads((snapshot_dir / "trace_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir), trust_remote_code=True)
    model = DebertaTraceSimple(cfg["backbone"])
    state = load_file(str(snapshot_dir / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model.eval()

    question = "Кто впервые показал синематограф в Париже?"
    chunks = [
        "Москва основана в 1147 году.",
        "22 марта 1895 года в Париже братьями Люмьер был впервые продемонстрирован их синематограф.",
        "Париж — столица Франции.",
    ]
    answer = "братьями Люмьер"

    out = score_chunks(question, chunks, answer, model, tokenizer, cfg["max_length"])
    thr_rel = cfg["thresholds"]["relevance"]
    thr_util = cfg["thresholds"]["utilization"]
    thr_adh = cfg["thresholds"]["adherence"]

    print(f"thresholds: relevance={{thr_rel}}  utilization={{thr_util}}  adherence={{thr_adh}}")
    print()
    print("=== Per-chunk scores (relevance + utilization) ===")
    for r in out["chunks"]:
        rel_m = r["rel_prob_mean"]
        util_m = r["util_prob_mean"]
        rel_pred = rel_m > thr_rel if rel_m == rel_m else False  # NaN-safe
        util_pred = util_m > thr_util if util_m == util_m else False
        print(f"  chunk {{r['chunk_id']}} | n_tokens={{r['n_tokens']:>3}} | "
              f"rel={{rel_m:.3f}} (pred_relevant={{rel_pred}}) | "
              f"util={{util_m:.3f}} (used_in_answer={{util_pred}})")

    print()
    print("=== Per-response score (adherence) ===")
    adh = out["mean_adherence_over_response"]
    print(f"  mean P(adherence over response tokens) = {{adh:.3f}}  "
          f"thr={{thr_adh:.2f}} -> answer_grounded={{adh > thr_adh}}")
    print()
    print("Hints:")
    print("  - relevance:   does this chunk contain information relevant to the question?")
    print("  - utilization: was this chunk actually used to produce the answer?")
    print("  - adherence:   are the response tokens grounded in the provided context?")


if __name__ == "__main__":
    main()
'''


# ----------------------------------------------------------------------------
# Model card
# ----------------------------------------------------------------------------

def _format_metrics_block(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "_(не сохранены в `result.json`)_"
    lines = ["| метрика | значение |", "| --- | ---: |"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def build_model_card(
    repo_id: str,
    backbone: str,
    architecture: str,
    max_length: int,
    thresholds: Dict[str, float],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    ood_metrics: Optional[Dict[str, Any]],
    license_id: str,
    loss_weights: Optional[Dict[str, Any]],
) -> str:
    ood_section = ""
    if ood_metrics:
        ood_section = (
            "## OOD-валидация (SberQuadQA, без подбора порогов)\n\n"
            f"{_format_metrics_block(ood_metrics)}\n\n"
            "Источник: `bearberry/sberquadqa`, прогон через тренировочные пороги "
            "(см. блок ниже). Никакая адаптация порогов на SberQuadQA не выполнялась.\n"
        )

    weights_str = json.dumps(loss_weights or {}, ensure_ascii=False)

    minimal_snippet = textwrap.dedent(f'''\
        from huggingface_hub import snapshot_download
        path = snapshot_download("{repo_id}")
        # см. examples/inference_minimal.py для полного примера
    ''')

    return textwrap.dedent(f"""\
        ---
        license: {license_id}
        language:
        - ru
        tags:
        - rag
        - rag-evaluation
        - trace
        - ragbench
        - token-classification
        base_model: {backbone}
        datasets:
        - CMCenjoyer/ragbench-ru
        ---

        # {repo_id.split('/')[-1]}

        Token-level TRACe-оценщик для русскоязычных RAG-систем (relevance / utilization / adherence).
        Адаптация фреймворка [TRACe из RAGBench](https://arxiv.org/abs/2407.11005) на русский язык.

        ## Архитектура

        - **Backbone:** [{backbone}](https://huggingface.co/{backbone})
        - **Голова:** {architecture} (paper-style: backbone → Dropout → 3×Linear на токен)
        - **Максимальная длина входа:** {max_length}
        - **Формат входа:** `[вопрос] [SEP] [документы] [SEP] [ответ]`
        - **Веса лосса при обучении:** `{weights_str}`

        ## Тренировочные данные

        [`CMCenjoyer/ragbench-ru`](https://huggingface.co/datasets/CMCenjoyer/ragbench-ru) — машинный
        перевод 7 поддатасетов RAGBench на русский (Qwen2.5-72B-Instruct).

        ## Метрики на validation (in-domain)

        {_format_metrics_block(val_metrics)}

        ## Метрики на test (in-domain)

        {_format_metrics_block(test_metrics)}

        {ood_section}
        ## Пороги классификации

        Подобраны на validation тренировочного датасета (`CMCenjoyer/ragbench-ru`),
        фиксируются в репозитории как часть релиза:

        ```json
        {json.dumps(thresholds, ensure_ascii=False, indent=2)}
        ```

        > **Внимание.** Пороги настроены под распределение скоров на RAGBench-RU.
        > При применении модели к данным из другого домена пороги следует
        > **калибровать заново** на валидационной подвыборке целевого домена.
        > Для threshold-free сравнения используйте ROC-AUC / PR-AUC по непрерывным `sigmoid`-скорам.

        ## Минимальный пример инференса

        ```python
        {minimal_snippet}
        ```

        Полные примеры:
        - [`examples/inference_minimal.py`](./examples/inference_minimal.py) — один пример,
          вывод среднего P(relevant) по токенам ответа.
        - [`examples/inference_with_chunks.py`](./examples/inference_with_chunks.py) — chunk-level
          скоринг с агрегацией token-level вероятностей.

        Зависимости: `torch`, `transformers`, `safetensors`, `huggingface_hub`.

        ## Ограничения

        - Модель обучена на **машинно-переведённых** данных — без ручной валидации
          качества перевода. Возможны систематические искажения в специализированных
          доменах (юриспруденция, медицина).
        - Эксперименты проведены с одним random seed; статистическая значимость
          различий между близкими конфигурациями не оценивалась.
        - Архитектура соответствует оригинальной статье RAGBench (paper-style),
          без собственных модификаций.
        - Пороги классификации **зависят от домена** — см. предупреждение выше.

        ## Цитирование

        ```bibtex
        @article{{friel2024ragbench,
          title={{RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems}},
          author={{Friel, Robert and Belyi, Masha and Sanyal, Atindriyo}},
          journal={{arXiv preprint arXiv:2407.11005}},
          year={{2024}}
        }}
        ```

        ## Связанные ресурсы

        - Тренировочный датасет: [`CMCenjoyer/ragbench-ru`](https://huggingface.co/datasets/CMCenjoyer/ragbench-ru)
        - OOD-датасет: [`bearberry/sberquadqa`](https://huggingface.co/datasets/bearberry/sberquadqa)
        - Исходный фреймворк (EN): [RAGBench / TRACe](https://arxiv.org/abs/2407.11005)
        """)


# ----------------------------------------------------------------------------
# Release builder
# ----------------------------------------------------------------------------

def _gather_ood_metrics(model_key: str) -> Optional[Dict[str, Any]]:
    """Try to find the matching SberQuadQA metrics_summary.json next to the
    OOD outputs. Returns a flat dict suitable for the model card."""
    # Map model_key -> external_eval folder slug used by score_dataset.py.
    slug_map = {
        "DeBERTa-v3-large (len=512)": "deberta-v3-large_len512__len_512",
        "ModernBERT (len=2048)":      "modernbert_len2048__len_2048",
        "BAAI-bge-m3 (len=1024)":     "baai-bge-m3_len1024__len_1024",
    }
    slug = slug_map.get(model_key)
    if slug is None:
        return None
    p = REPO_ROOT / "external_eval" / "sberquadqa" / "outputs" / slug / "metrics_summary.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    keep = ("accuracy", "precision", "recall", "f1", "mcc",
            "roc_auc_mean", "pr_auc_mean", "top1_acc", "topk_recall",
            "n_examples_processed", "n_examples_skipped")
    return {k: data[k] for k in keep if k in data}


def build_release_folder(
    model_key: str,
    repo_id: str,
    out_dir: Path,
    license_id: str,
) -> Dict[str, Any]:
    """Assemble the folder we will upload. Returns a manifest dict."""
    pre = preflight_check(model_key)
    if not pre.ok:
        raise SystemExit(
            f"Preflight failed for '{model_key}': {pre.message}\n"
            "Cannot publish a checkpoint that does not exist on disk."
        )

    state_path = Path(pre.state_path)
    state_dir = state_path.parent
    backbone = pre.checkpoint_dir
    entry = MODEL_REGISTRY[model_key]
    max_length = int(entry["max_length"])
    thresholds = pre.thresholds

    # Read result.json for metrics + loss weights (optional).
    val_metrics: Dict[str, Any] = {}
    test_metrics: Dict[str, Any] = {}
    loss_weights: Optional[Dict[str, Any]] = None
    result_data: Dict[str, Any] = {}
    result_path = state_dir / "result.json"
    if result_path.exists():
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
            val_metrics = result_data.get("val_metrics") or {}
            test_metrics = result_data.get("test_metrics") or {}
            cfg_dict = result_data.get("cfg_dict") or {}
            train_cfg = (cfg_dict.get("train") or {}) if isinstance(cfg_dict, dict) else {}
            loss_weights = {
                "relevance":   train_cfg.get("loss_weight_relevance"),
                "utilization": train_cfg.get("loss_weight_utilization"),
                "adherence":   train_cfg.get("loss_weight_adherence"),
            }
            if all(v is None for v in loss_weights.values()):
                loss_weights = None
        except Exception as e:  # noqa: BLE001
            print(f"[publish] warning: could not parse result.json: {e}")

    # Load + verify state_dict against the inferred architecture.
    print(f"[publish] loading state_dict from {state_path}")
    state_dict = _load_state_dict(state_path)
    architecture = _detect_architecture(state_dict)
    print(f"[publish] detected architecture: {architecture}")

    print(f"[publish] instantiating backbone '{backbone}' to verify keys")
    model = _build_model(backbone, architecture)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Keys for the *backbone* sometimes appear as 'missing' simply because
        # they were initialised on the fly from the HF backbone. Anything in
        # the state_dict that does NOT match the model is a real problem.
        backbone_missing = [k for k in missing if not k.startswith("base.")]
        if backbone_missing or unexpected:
            print("[publish] WARNING — strict load failed:")
            print(f"  unexpected: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
            print(f"  missing (non-backbone): {backbone_missing[:10]}")
            raise SystemExit(
                "State_dict does not match the inferred architecture. "
                "Refusing to publish a checkpoint that won't load cleanly."
            )

    # Build the folder.
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)

    # 1. weights -> safetensors
    safetensors_path = out_dir / "model.safetensors"
    print(f"[publish] writing safetensors -> {safetensors_path}")
    _safetensors_save(state_dict, safetensors_path)

    # 2. tokenizer
    tok_status = _copy_tokenizer(state_dir, backbone, out_dir)
    print(f"[publish] tokenizer: {tok_status}")

    # 3. backbone config
    _copy_backbone_config(backbone, out_dir)
    print(f"[publish] wrote config.json from backbone")

    # 4. trace_config + thresholds
    ood_metrics = _gather_ood_metrics(model_key)
    trace_config = {
        "trace_format_version": 1,
        "trace_architecture": architecture,
        "backbone": backbone,
        "max_length": max_length,
        "language": "ru",
        "trained_on": "CMCenjoyer/ragbench-ru",
        "loss_weights": loss_weights,
        "thresholds": thresholds,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ood_metrics": {"sberquadqa": ood_metrics} if ood_metrics else None,
        "input_format": "[question] [SEP] [documents] [SEP] [response]",
        "model_key_in_repo": model_key,
    }
    (out_dir / "trace_config.json").write_text(
        json.dumps(trace_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 5. README
    card = build_model_card(
        repo_id=repo_id,
        backbone=backbone,
        architecture=architecture,
        max_length=max_length,
        thresholds=thresholds,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        ood_metrics=ood_metrics,
        license_id=license_id,
        loss_weights=loss_weights,
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")

    # 6. examples
    (out_dir / "examples" / "inference_minimal.py").write_text(
        EXAMPLE_INFERENCE_MINIMAL.format(repo_id=repo_id), encoding="utf-8"
    )
    (out_dir / "examples" / "inference_with_chunks.py").write_text(
        EXAMPLE_INFERENCE_WITH_CHUNKS.format(repo_id=repo_id), encoding="utf-8"
    )

    manifest = {
        "repo_id": repo_id,
        "out_dir": str(out_dir),
        "files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
        "size_bytes_total": sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()),
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[publish] release built at {out_dir}")
    print(f"[publish] total size: {manifest['size_bytes_total']/1e6:.1f} MB")
    print(f"[publish] files: {manifest['files']}")
    return manifest


def push_to_hub(out_dir: Path, repo_id: str, private: bool, commit_message: str):
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    print(f"[publish] creating/ensuring repo {repo_id} (private={private})")
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model", token=token)

    print(f"[publish] uploading folder {out_dir} -> {repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(out_dir),
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=["_manifest.json"],
    )
    print(f"[publish] done. https://huggingface.co/{repo_id}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_key", required=True,
                    help=f"One of {list(MODEL_REGISTRY)} from external_eval/sberquadqa/_common.py")
    ap.add_argument("--repo_id", required=True,
                    help="HF repo id, e.g. CMCenjoyer/ru-trace-modernbert-2048")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Where to assemble the release. Default: <state_dir>/hf_release")
    ap.add_argument("--license", default="mit", choices=["mit", "apache-2.0", "cc-by-4.0"],
                    help="License id for the model card. Default: mit")
    ap.add_argument("--private", action="store_true", help="Create the HF repo as private")
    ap.add_argument("--dry_run", action="store_true",
                    help="Build the folder locally but DO NOT push to the Hub")
    ap.add_argument("--commit_message", default="Initial release",
                    help="Commit message used for the upload")
    return ap.parse_args()


def main():
    args = parse_args()
    pre = preflight_check(args.model_key)
    state_dir = Path(pre.state_path).parent if pre.state_path else Path(".")
    out_dir = args.out_dir or (state_dir / "hf_release")

    manifest = build_release_folder(
        model_key=args.model_key,
        repo_id=args.repo_id,
        out_dir=out_dir,
        license_id=args.license,
    )

    if args.dry_run:
        print("[publish] --dry_run: skipping push to Hub.")
        print(f"[publish] inspect the release at: {out_dir}")
        return

    if not (os.environ.get("HF_TOKEN") or _has_hf_login()):
        raise SystemExit(
            "No HuggingFace credentials found. Run `huggingface-cli login` "
            "or set HF_TOKEN env var before publishing."
        )
    push_to_hub(
        out_dir=out_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )


def _has_hf_login() -> bool:
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token() is not None
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
