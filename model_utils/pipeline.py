import dataclasses
import os
from pathlib import Path

import torch
from torch.amp import GradScaler
from transformers import AutoTokenizer

from config import ProjectConfig
from data import (
    make_raw_combined,
    tokenize_from_raw,
    make_dataloaders,
    save_dataset,
    load_dataset,
    safe_model_tag,
    raw_cache_path,
    tokenized_cache_path,
)
from metrics import (
    compute_classification_metrics_stream,
    compute_metrics_with_thresholds,
    tune_thresholds_on_val,
)
from models import DebertaTrace
from train_loop import run_training_loop
from results import ExperimentResult, save_result


def _validate_raw_data(combined_raw, data_cfg) -> None:
    """Validate raw data BEFORE tokenization: check key formats and dataset-level coverage.

    Three-level check per split:
      1. Examples with non-empty relevant/utilized keys
      2. Examples where at least one document key matches a relevant/utilized key
      3. Estimated positive sentence ratio (would-be labels)
    """
    def _match(k, keys):
        return k in keys or any(x.startswith(k) for x in keys)

    print("\n[pipeline] ── Raw data validation ──────────────────────────────────")
    for split_name in ("train", "validation", "test"):
        ds = combined_raw.get(split_name)
        if ds is None or len(ds) == 0:
            print(f"  {split_name}: empty or missing")
            continue

        ex0 = ds[0]
        rel_keys_0 = set(ex0.get(data_cfg.relevant_keys_field, []))
        util_keys_0 = set(ex0.get(data_cfg.utilized_keys_field, []))
        doc_keys_0 = []
        for doc in ex0.get(data_cfg.documents_field, []):
            for item in doc:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    doc_keys_0.append(item[0])

        print(f"\n  [{split_name}] example 0 spot-check:")
        print(f"    relevant_keys ({len(rel_keys_0)}): {sorted(rel_keys_0)[:5]}")
        print(f"    utilized_keys ({len(util_keys_0)}): {sorted(util_keys_0)[:5]}")
        print(f"    document_keys ({len(doc_keys_0)}): {doc_keys_0[:5]}")
        matched_0 = sum(1 for k in doc_keys_0 if _match(k, rel_keys_0))
        print(f"    matched rel (prefix): {matched_0}/{len(doc_keys_0)}")
        if len(doc_keys_0) > 0 and matched_0 == 0 and len(rel_keys_0) > 0:
            print(f"    *** WARNING: KEY FORMAT MISMATCH in example 0! ***")
            print(f"    *** doc key sample: {doc_keys_0[0]!r}")
            print(f"    *** rel key sample: {sorted(rel_keys_0)[0]!r}")

        n_total = len(ds)
        n_with_rel_keys = 0
        n_with_util_keys = 0
        n_with_rel_match = 0
        n_with_util_match = 0
        total_sentences = 0
        total_rel_sentences = 0
        total_util_sentences = 0

        for ex in ds:
            rel_keys = set(ex.get(data_cfg.relevant_keys_field, []))
            util_keys = set(ex.get(data_cfg.utilized_keys_field, []))

            if len(rel_keys) > 0:
                n_with_rel_keys += 1
            if len(util_keys) > 0:
                n_with_util_keys += 1

            doc_keys = []
            for doc_item in ex.get(data_cfg.documents_field, []):
                for item in doc_item:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        doc_keys.append(item[0])

            total_sentences += len(doc_keys)
            rel_matched = sum(1 for k in doc_keys if _match(k, rel_keys))
            util_matched = sum(1 for k in doc_keys if _match(k, util_keys))
            total_rel_sentences += rel_matched
            total_util_sentences += util_matched
            if rel_matched > 0:
                n_with_rel_match += 1
            if util_matched > 0:
                n_with_util_match += 1

        print(f"\n  [{split_name}] dataset-level coverage (n={n_total}):")
        print(f"    examples with rel_keys:   {n_with_rel_keys}/{n_total} ({n_with_rel_keys/max(n_total,1):.1%})")
        print(f"    examples with util_keys:  {n_with_util_keys}/{n_total} ({n_with_util_keys/max(n_total,1):.1%})")
        print(f"    examples with rel_match:  {n_with_rel_match}/{n_total} ({n_with_rel_match/max(n_total,1):.1%})")
        print(f"    examples with util_match: {n_with_util_match}/{n_total} ({n_with_util_match/max(n_total,1):.1%})")
        print(f"    positive rel sentences:   {total_rel_sentences}/{total_sentences} ({total_rel_sentences/max(total_sentences,1):.1%})")
        print(f"    positive util sentences:  {total_util_sentences}/{total_sentences} ({total_util_sentences/max(total_sentences,1):.1%})")

        if n_with_rel_keys == 0:
            print(f"    *** DIAGNOSIS: Dataset has NO relevant_keys annotations at all ***")
        elif n_with_rel_match == 0:
            print(f"    *** DIAGNOSIS: KEY FORMAT MISMATCH — keys exist but never match document keys ***")
    print("[pipeline] ─────────────────────────────────────────────────────────\n")


def _log_dataset_stats(tokenized, tokenizer, output_dir: Path) -> None:
    """Print label positive-rate and length stats; dump 3 examples to debug_examples.txt."""
    import numpy as np

    print("\n[pipeline] ── Dataset diagnostics ──────────────────────────────────")
    for split_name, ds in tokenized.items():
        if len(ds) == 0:
            print(f"  {split_name}: empty")
            continue

        lengths = [len(ex["input_ids"]) for ex in ds]
        ctx_pos_rel, ctx_pos_util, resp_pos_adh, ctx_total, resp_total = 0, 0, 0, 0, 0
        for ex in ds:
            ctx  = [bool(m) for m in ex["context_mask"]]
            resp = [bool(m) for m in ex["response_mask"]]
            lbl_r = ex["labels_relevance"]
            lbl_u = ex["labels_utilization"]
            lbl_a = ex["labels_adherence"]
            ctx_total  += sum(ctx)
            resp_total += sum(resp)
            ctx_pos_rel  += sum(lbl_r[i] == 1.0 for i, c in enumerate(ctx)  if c)
            ctx_pos_util += sum(lbl_u[i] == 1.0 for i, c in enumerate(ctx)  if c)
            resp_pos_adh += sum(lbl_a[i] == 1.0 for i, r in enumerate(resp) if r)

        rate_rel  = ctx_pos_rel  / max(ctx_total,  1)
        rate_util = ctx_pos_util / max(ctx_total,  1)
        rate_adh  = resp_pos_adh / max(resp_total, 1)
        print(
            f"  {split_name:10s} n={len(ds):6d} "
            f"len=[{min(lengths)},{int(np.mean(lengths))},{max(lengths)}] "
            f"pos_rel={rate_rel:.1%} pos_util={rate_util:.1%} pos_adh={rate_adh:.1%}"
        )

    debug_path = output_dir / "debug_examples.txt"
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            split_ds = tokenized.get("train", list(tokenized.values())[0])
            for ex_idx in range(min(5, len(split_ds))):
                ex = split_ds[ex_idx]
                ids = list(ex["input_ids"])
                tokens = tokenizer.convert_ids_to_tokens(ids)
                ctx  = list(ex["context_mask"])
                resp = list(ex["response_mask"])
                lbl_r = list(ex["labels_relevance"])
                lbl_u = list(ex["labels_utilization"])
                lbl_a = list(ex["labels_adherence"])

                n_ctx = sum(ctx)
                n_pos_r = sum(1 for i, c in enumerate(ctx) if c and lbl_r[i] == 1.0)
                n_pos_u = sum(1 for i, c in enumerate(ctx) if c and lbl_u[i] == 1.0)

                f.write(f"\n=== Example {ex_idx} (len={len(ids)}) ===\n")
                f.write(f"  ctx_tokens={n_ctx}  pos_rel={n_pos_r} ({n_pos_r/max(n_ctx,1):.1%})  pos_util={n_pos_u} ({n_pos_u/max(n_ctx,1):.1%})\n")
                f.write(f"{'Token':20} | ctx | resp | rel | util | adh\n")
                f.write("-" * 60 + "\n")
                for tok, c, r, lr, lu, la in zip(tokens, ctx, resp, lbl_r, lbl_u, lbl_a):
                    marker = ""
                    if lr == 1.0 or lu == 1.0:
                        marker = "  ← POS"
                    f.write(f"{tok:20} |  {int(c)}  |  {int(r)}   | {lr:.1f} | {lu:.1f}  | {la:.1f}{marker}\n")
        print(f"[pipeline] Debug examples written to: {debug_path}")
    except Exception as e:
        print(f"[pipeline] Warning: could not write debug examples: {e}")
    print("[pipeline] ─────────────────────────────────────────────────────────\n")


def train(cfg: ProjectConfig, rewrite_dataset: bool = False) -> ExperimentResult:
    """Full training pipeline.  Returns an ExperimentResult and writes result.json.

    Args:
        cfg: A ProjectConfig instance.  Use cfg.with_overrides(...) to create
             per-run variants without mutating the base config.
        rewrite_dataset: Force re-tokenization even if a cached version exists.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[pipeline] Device:", device)
    print("[pipeline] cwd:", os.getcwd())
    print("[pipeline] output_dir:", cfg.train.output_dir)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_cache_path(cfg.train.output_dir)
    if raw_path.exists():
        print(f"[pipeline] DATA SOURCE: RAW from cache ({raw_path})")
        combined_raw = load_dataset(str(raw_path))
    else:
        print("[pipeline] DATA SOURCE: RAW freshly built")
        combined_raw = make_raw_combined(cfg.data)
        print(f"[pipeline] Saving RAW dataset to {raw_path}")
        save_dataset(combined_raw, str(raw_path))

    _validate_raw_data(combined_raw, cfg.data)

    model_name = cfg.model.pretrained_name
    tok_path = tokenized_cache_path(cfg.train.output_dir, model_name, cfg.data.max_length)

    if cfg.train.save_datasets and tok_path.exists() and not rewrite_dataset:
        print(f"[pipeline] DATA SOURCE: TOKENIZED from cache ({tok_path})")
        tokenized = load_dataset(str(tok_path))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        effective_max_len = cfg.data.max_length
    else:
        print(f"[pipeline] DATA SOURCE: TOKENIZED freshly built for model={model_name}")
        tokenized, tokenizer, effective_max_len = tokenize_from_raw(
            combined_raw, cfg.data, model_name, debug_dir=str(output_dir)
        )
        print("\n[pipeline][TOKENIZED dataset sizes]")
        print(f"  model      : {model_name}")
        print(f"  max_length : {effective_max_len}")
        for split, ds in tokenized.items():
            print(f"  {split:10s}: {len(ds):7d}")

        if cfg.train.save_datasets:
            print(f"[pipeline] Saving TOKENIZED dataset to {tok_path}")
            save_dataset(tokenized, str(tok_path))

    train_sizes = {split: len(ds) for split, ds in tokenized.items()}

    _log_dataset_stats(tokenized, tokenizer, output_dir)


    train_ds = tokenized["train"]
    sample_size = min(200, len(train_ds))
    pos_rel_total = 0
    pos_util_total = 0
    for j in range(sample_size):
        ex = train_ds[j]
        ctx = ex["context_mask"]
        lbl_r = ex["labels_relevance"]
        lbl_u = ex["labels_utilization"]
        for i, c in enumerate(ctx):
            if c:
                if lbl_r[i] == 1.0:
                    pos_rel_total += 1
                if lbl_u[i] == 1.0:
                    pos_util_total += 1
    if pos_rel_total == 0 and pos_util_total == 0:
        raise ValueError(
            f"[pipeline] FATAL: Zero positive relevance AND utilization labels "
            f"in first {sample_size} training examples. "
            f"Most likely cause: key format mismatch between "
            f"'{cfg.data.documents_field}' and '{cfg.data.relevant_keys_field}'. "
            f"Check build_example_debug.txt and _validate_raw_data output above. "
            f"Try rewrite_dataset=True to force re-tokenization."
        )

    effective_batch = cfg.train.resolve_train_batch_size(effective_max_len)
    if effective_batch != cfg.train.train_batch_size:
        print(f"[pipeline] Resolved train_batch_size={effective_batch} for max_length={effective_max_len}")
    train_cfg_for_loaders = dataclasses.replace(cfg.train, train_batch_size=effective_batch)
    train_loader, val_loader, test_loader = make_dataloaders(tokenized, tokenizer, train_cfg_for_loaders)

    model = DebertaTrace(model_name, use_complex=cfg.model.use_complex_model).to(device)
    print(f"[pipeline] Model architecture: {'complex' if cfg.model.use_complex_model else 'simple (paper-style)'}")

    if cfg.train.backbone_lr is not None and cfg.train.head_lr is not None:
        backbone_params = [p for n, p in model.named_parameters() if n.startswith("base.")]
        head_params     = [p for n, p in model.named_parameters() if not n.startswith("base.")]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.train.backbone_lr},
                {"params": head_params,     "lr": cfg.train.head_lr},
            ],
            weight_decay=cfg.train.weight_decay,
        )
        print(f"[pipeline] Split LR: backbone={cfg.train.backbone_lr}, head={cfg.train.head_lr}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        print(f"[pipeline] Single LR: {cfg.train.learning_rate}")
    use_amp = bool(cfg.model.use_fp16 and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    scheduler = None
    if cfg.train.warmup_ratio and cfg.train.warmup_ratio > 0.0:
        from transformers import get_linear_schedule_with_warmup
        total_steps = len(train_loader) * cfg.train.num_epochs
        warmup_steps = int(cfg.train.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(f"[pipeline] LR scheduler: linear warmup {warmup_steps}/{total_steps} steps (ratio={cfg.train.warmup_ratio})")

    best_state, val_metrics, best_val_f1 = run_training_loop(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=cfg.train,
        device=device,
        use_amp=use_amp,
        scheduler=scheduler,
    )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    def _run_eval_loop(loader):
        logits_list, labels_list, masks_list = [], [], []
        with torch.no_grad():
            for batch in loader:
                inputs = {
                    "input_ids":      batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                lbls  = {k: v for k, v in batch.items() if k.startswith("labels_")}
                msks  = {k: v for k, v in batch.items() if k.endswith("_mask")}
                out = model(**inputs)
                logits_list.append({
                    "logits_relevance":   out["logits_relevance"].cpu(),
                    "logits_utilization": out["logits_utilization"].cpu(),
                    "logits_adherence":   out["logits_adherence"].cpu(),
                })
                labels_list.append({
                    "labels_relevance":   lbls["labels_relevance"].cpu(),
                    "labels_utilization": lbls["labels_utilization"].cpu(),
                    "labels_adherence":   lbls["labels_adherence"].cpu(),
                })
                masks_list.append({
                    "context_mask":  msks["context_mask"].cpu(),
                    "response_mask": msks["response_mask"].cpu(),
                })
        return logits_list, labels_list, masks_list

    print("[pipeline] Re-evaluating validation set with best checkpoint...")
    val_logits, val_labels, val_masks = _run_eval_loop(val_loader)

    print("[pipeline] Tuning per-target thresholds on validation set...")
    thresholds = tune_thresholds_on_val(val_logits, val_labels, val_masks)

    val_metrics = compute_metrics_with_thresholds(val_logits, val_labels, val_masks, thresholds)
    best_val_f1 = val_metrics.get("relevance_f1", best_val_f1)
    print("\n[pipeline] Val metrics (best checkpoint, tuned thresholds):")
    for name, v in val_metrics.items():
        print(f"  {name}: {v:.4f}")

    print("[pipeline] Running test evaluation with tuned thresholds...")
    test_logits_batches, test_labels_batches, test_masks_batches = _run_eval_loop(test_loader)

    test_metrics = compute_metrics_with_thresholds(
        test_logits_batches, test_labels_batches, test_masks_batches, thresholds
    )
    print("\n[pipeline] Test metrics (best checkpoint, tuned thresholds):")
    for name, v in test_metrics.items():
        print(f"  {name}: {v:.4f}")

    model_path = output_dir / "model.pt"

    state_to_save = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_to_save, str(model_path))
    tokenizer.save_pretrained(str(output_dir))
    print("[pipeline] Saved model to:", model_path)
    print("[pipeline] Saved tokenizer to:", output_dir)

    result = ExperimentResult(
        model_name=model_name,
        max_length=effective_max_len,
        output_dir=str(output_dir),
        val_metrics=val_metrics,
        best_val_f1=best_val_f1,
        train_sizes=train_sizes,
        cfg_dict=cfg.to_dict(),
        test_metrics=test_metrics,
        thresholds=thresholds,
    )
    save_result(result, str(output_dir / "result.json"))
    return result
