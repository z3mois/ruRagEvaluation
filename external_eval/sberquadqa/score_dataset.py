from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ._common import (
    MODEL_REGISTRY,
    aggregate_chunk_scores,
    load_trace_model,
    preflight_check,
    preprocess_one_with_chunk_ids,
    run_inference_single,
    safe_model_tag,
)

DEFAULT_DATA = Path(__file__).resolve().parent / "cache" / "sberquadqa_prepared.parquet"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "outputs"


def auto_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--limit", type=int, default=None, help="Process only first N examples (debug).")
    ap.add_argument("--truncate", action="store_true",
                    help="Truncate chunks from the end if example doesn't fit (default: skip).")
    ap.add_argument("--response_mode", default="gold_answer", choices=["gold_answer"],
                    help="How response_ru is built. Currently only 'gold_answer' (= answers[0]).")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Defaults to outputs/<safe_model_tag>.")
    args = ap.parse_args()

    device = auto_device() if args.device == "auto" else args.device
    print(f"[score] device={device}, model_key={args.model_key}")

    pre = preflight_check(args.model_key)
    print(f"[score] preflight: {pre.message}")
    if not pre.ok:
        raise SystemExit(
            f"Cannot proceed: {pre.message}.\n"
            f"Expected weights at: {pre.state_path}\n"
            "This script does NOT download or train models — provide the checkpoint manually."
        )

    entry = MODEL_REGISTRY[args.model_key]
    max_length = int(entry["max_length"])
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / safe_model_tag(args.model_key, max_length))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        raise SystemExit(
            f"Prepared data not found: {args.data}. "
            "Run prepare_dataset.py first."
        )

    import pandas as pd
    df = pd.read_parquet(args.data)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"[score] examples to process: {len(df)}")

    print(f"[score] loading model + tokenizer...")
    tokenizer, model, thresholds, threshold_source, _ = load_trace_model(args.model_key, device)
    print(f"[score] thresholds={thresholds} (source={threshold_source})")

    chunk_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    n_skipped = 0
    n_truncated = 0
    n_processed = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(df.itertuples(index=False), total=len(df), desc=f"score[{args.model_key}]")
    except Exception:
        iterator = df.itertuples(index=False)

    t0 = time.time()
    for row in iterator:
        ex = row._asdict() if hasattr(row, "_asdict") else dict(row)
        gt_per_chunk = list(ex["gt_is_relevant_per_chunk"])
        docs = ex["documents_sentences_ru"]

        example_payload = {
            "question_ru": ex["question_ru"],
            "response_ru": ex["response_ru"],
            "documents_sentences_ru": docs,
            "all_relevant_sentence_keys": [],
            "all_utilized_sentence_keys": [],
            "adherence_score": 0.0,
        }
        batch = preprocess_one_with_chunk_ids(
            example_payload, tokenizer, max_length, truncate=args.truncate
        )
        if batch is None:
            n_skipped += 1
            example_rows.append({
                "example_id": ex["orig_id"],
                "n_chunks": int(ex["n_chunks"]),
                "n_relevant_gt": int(sum(gt_per_chunk)),
                "pred_top1_idx": -1,
                "pred_topk_idxs": [],
                "mean_resp_adherence_prob": float("nan"),
                "is_answerable": bool(ex["is_answerable"]),
                "truncated": False,
                "n_relevant_dropped": 0,
                "skipped": True,
            })
            continue

        if batch["n_chunks_dropped"] > 0:
            n_truncated += 1

        res = run_inference_single(model, batch, device)
        chunk_id_per_token = batch["chunk_id_per_token"].cpu().numpy()
        n_used = batch["n_chunks_used"]
        per_chunk = aggregate_chunk_scores(res["probs"], chunk_id_per_token, n_used)

        resp_mask = res["masks"]["response"]
        if resp_mask.any():
            mean_adh = float(np.asarray(res["probs"]["adherence"])[resp_mask].mean())
        else:
            mean_adh = float("nan")

        n_relevant_dropped = sum(1 for i, v in enumerate(gt_per_chunk) if v and i >= n_used)
        n_relevant_gt = int(sum(gt_per_chunk))

        chunk_text_by_id = {i: docs[0][i][1] for i in range(n_used)}
        for r in per_chunk:
            cid = r["chunk_id"]
            gt = bool(gt_per_chunk[cid]) if cid < len(gt_per_chunk) else False
            chunk_rows.append({
                "example_id": ex["orig_id"],
                "chunk_id": cid,
                "chunk_key": f"chunk_{cid:04d}",
                "chunk_text": chunk_text_by_id.get(cid, ""),
                "gt_is_relevant": gt,
                "n_tokens": r["n_tokens"],
                "rel_prob_mean": r["rel_prob_mean"],
                "rel_prob_max": r["rel_prob_max"],
                "util_prob_mean": r["util_prob_mean"],
                "util_prob_max": r["util_prob_max"],
                "pred_relevant": bool(r["rel_prob_mean"] > thresholds["relevance"]),
                "n_chunks_in_example": n_used,
                "truncated": batch["n_chunks_dropped"] > 0,
                "model_key": args.model_key,
                "max_length": max_length,
            })

        scores = np.array([r["rel_prob_mean"] for r in per_chunk]) if per_chunk else np.array([])
        if scores.size > 0:
            order = np.argsort(-scores)
            top1_idx = int(order[0])
            k = max(1, n_relevant_gt)
            topk_idxs = order[:k].tolist()
        else:
            top1_idx = -1
            topk_idxs = []

        example_rows.append({
            "example_id": ex["orig_id"],
            "n_chunks": int(ex["n_chunks"]),
            "n_relevant_gt": n_relevant_gt,
            "pred_top1_idx": top1_idx,
            "pred_topk_idxs": topk_idxs,
            "mean_resp_adherence_prob": mean_adh,
            "is_answerable": bool(ex["is_answerable"]),
            "truncated": batch["n_chunks_dropped"] > 0,
            "n_relevant_dropped": int(n_relevant_dropped),
            "skipped": False,
        })
        n_processed += 1

    elapsed = time.time() - t0
    print(f"[score] processed={n_processed}, skipped={n_skipped}, truncated={n_truncated}, "
          f"time={elapsed:.1f}s")


    chunks_df = pd.DataFrame(chunk_rows)
    examples_df = pd.DataFrame(example_rows)
    chunks_path = out_dir / "chunks_scored.parquet"
    examples_path = out_dir / "examples_scored.parquet"
    chunks_df.to_parquet(chunks_path, index=False)
    examples_df.to_parquet(examples_path, index=False)
    print(f"[score] wrote {len(chunks_df)} chunk rows -> {chunks_path}")
    print(f"[score] wrote {len(examples_df)} example rows -> {examples_path}")

    run_meta = {
        "model_key": args.model_key,
        "checkpoint_dir": entry["checkpoint_dir"],
        "state_path": entry["state_path"],
        "max_length": max_length,
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "response_mode": args.response_mode,
        "truncate_enabled": bool(args.truncate),
        "device": device,
        "data_path": str(args.data),
        "n_examples_input": int(len(df)),
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "n_truncated": n_truncated,
        "elapsed_sec": round(elapsed, 2),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[score] wrote run_meta -> {out_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
