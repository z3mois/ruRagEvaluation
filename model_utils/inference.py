from pathlib import Path
from typing import Optional
import os

import torch
from transformers import AutoTokenizer

from config import cfg as _default_cfg, DataConfig, TrainingConfig
from models import DebertaTrace
from data import (
    load_dataset,
    save_dataset,
    tokenize_from_raw,
    safe_model_tag,
    make_dataloaders,
    tokenized_cache_path,
)
from metrics import compute_classification_metrics_stream


def evaluate_split(
    split: str = "validation",
    model_name: str = None,
    model_dir: Optional[str] = None,
    raw_path: Optional[str] = None,
    threshold: float = 0.5,
    reuse_tokenized: bool = True,
    tokenized_root: Optional[str] = None,
    data_cfg: Optional[DataConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
):
    """Evaluate a saved model checkpoint on a given dataset split.

    Args:
        split: One of "train", "validation", "test".
        model_name: HuggingFace model name.  Defaults to cfg.model.pretrained_name.
        model_dir: Directory containing model.pt (and optionally the tokenizer).
                   Defaults to cfg.train.output_dir.
        raw_path: Path to the raw combined dataset.  Defaults to <model_dir>/raw/combined.
        threshold: Classification threshold for compute_classification_metrics_stream.
        reuse_tokenized: Load cached tokenized dataset if available.
        tokenized_root: Root for tokenized cache.  Defaults to <model_dir>/tokenized.
        data_cfg: DataConfig to use for tokenization.  Defaults to cfg.data.
        train_cfg: TrainingConfig for batch sizes.  Defaults to cfg.train.
    """
    assert split in ("train", "validation", "test"), f"Unknown split: {split}"

    if model_name is None:
        model_name = _default_cfg.model.pretrained_name
    if model_dir is None:
        model_dir = _default_cfg.train.output_dir
    if data_cfg is None:
        data_cfg = _default_cfg.data
    if train_cfg is None:
        train_cfg = _default_cfg.train

    model_dir = Path(model_dir)

    if raw_path is None:
        raw_path = str(model_dir / "raw" / "combined")
    if tokenized_root is None:
        tokenized_root = str(model_dir / "tokenized")

    combined_raw = load_dataset(raw_path)
    print("\n[inference][RAW dataset sizes]")
    print(f"  raw_path: {raw_path}")
    for s, ds in combined_raw.items():
        print(f"  {s:10s}: {len(ds):7d}")

    model_tag = safe_model_tag(model_name)
    tok_path = Path(tokenized_root) / model_tag

    if reuse_tokenized and tok_path.exists():
        tokenized = load_dataset(str(tok_path))
        if (model_dir / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenized, tokenizer, _ = tokenize_from_raw(combined_raw, data_cfg, model_name)
        if reuse_tokenized:
            save_dataset(tokenized, str(tok_path))

    train_loader, val_loader, test_loader = make_dataloaders(tokenized, tokenizer, train_cfg)
    loader = {"train": train_loader, "validation": val_loader, "test": test_loader}[split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaTrace(model_name).to(device)

    weight_path = model_dir / "model.pt"
    print(os.listdir(model_dir))
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing weights: {weight_path}")

    try:
        state = torch.load(str(weight_path), map_location=device, weights_only=True)
    except Exception:
        state = torch.load(str(weight_path), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    all_logits, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labels = {k: v for k, v in batch.items() if k.startswith("labels_")}
            masks  = {k: v for k, v in batch.items() if k.endswith("_mask")}

            out = model(**inputs)

            all_logits.append({
                "logits_relevance":   out["logits_relevance"].cpu(),
                "logits_utilization": out["logits_utilization"].cpu(),
                "logits_adherence":   out["logits_adherence"].cpu(),
            })
            all_labels.append({
                "labels_relevance":   labels["labels_relevance"].cpu(),
                "labels_utilization": labels["labels_utilization"].cpu(),
                "labels_adherence":   labels["labels_adherence"].cpu(),
            })
            all_masks.append({
                "context_mask":  masks["context_mask"].cpu(),
                "response_mask": masks["response_mask"].cpu(),
            })

    metrics = compute_classification_metrics_stream(
        all_logits, all_labels, all_masks, threshold=threshold
    )
    return metrics
