from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
DEFAULT_OUT = Path(__file__).resolve().parent / "cache" / "sberquadqa_prepared.parquet"


def map_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    context = ex["context"] or []
    answers = ex.get("answers") or []
    response = answers[0] if answers else ""

    documents_sentences_ru = [
        [[f"chunk_{i:04d}", c["chunk"]] for i, c in enumerate(context)]
    ]
    gt_is_relevant_per_chunk = [bool(c["is_relevant"]) for c in context]
    gt_relevant_idx_list = [i for i, v in enumerate(gt_is_relevant_per_chunk) if v]

    metadata = ex.get("metadata") or {}
    is_answerable = bool(metadata.get("is_answerable", False))

    return {
        "question_ru": ex["question"],
        "response_ru": response,
        "documents_sentences_ru": documents_sentences_ru,
        "all_relevant_sentence_keys": [],
        "all_utilized_sentence_keys": [],
        "adherence_score": 0.0,

        "orig_id": ex["id"],
        "n_chunks": len(context),
        "gt_is_relevant_per_chunk": gt_is_relevant_per_chunk,
        "gt_relevant_idx_list": gt_relevant_idx_list,
        "is_answerable": is_answerable,
        "answers_all": list(answers),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_examples", default="5000",
                    help="Number of examples to keep (int or 'all'). Default: 5000.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_answerable", action="store_true",
                    help="Drop examples with metadata.is_answerable == False.")
    ap.add_argument("--no_require_positive", action="store_true",
                    help="Do NOT require at least one is_relevant=True chunk (default: require).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] loading bearberry/sberquadqa (split=train)...")
    from datasets import load_dataset
    ds = load_dataset("bearberry/sberquadqa", split="train")
    print(f"[prepare] raw size: {len(ds)}")

    require_positive = not args.no_require_positive

    mapped: List[Dict[str, Any]] = []
    n_dropped_no_pos = 0
    n_dropped_unanswerable = 0
    for ex in ds:
        m = map_example(ex)
        if require_positive and not any(m["gt_is_relevant_per_chunk"]):
            n_dropped_no_pos += 1
            continue
        if args.require_answerable and not m["is_answerable"]:
            n_dropped_unanswerable += 1
            continue
        mapped.append(m)

    print(f"[prepare] after filters: {len(mapped)} "
          f"(dropped no-positive={n_dropped_no_pos}, unanswerable={n_dropped_unanswerable})")

    if args.n_examples != "all":
        n = int(args.n_examples)
        if n < len(mapped):
            random.Random(args.seed).shuffle(mapped)
            mapped = mapped[:n]
            print(f"[prepare] sampled to {len(mapped)} (seed={args.seed})")


    df = pd.DataFrame(mapped)
    df.to_parquet(args.out, index=False)
    print(f"[prepare] wrote {len(df)} rows -> {args.out}")

    meta = {
        "source": "bearberry/sberquadqa",
        "split": "train",
        "n_raw": len(ds),
        "n_kept": int(len(df)),
        "n_dropped_no_pos": int(n_dropped_no_pos),
        "n_dropped_unanswerable": int(n_dropped_unanswerable),
        "require_positive": bool(require_positive),
        "require_answerable": bool(args.require_answerable),
        "seed": int(args.seed),
        "n_examples_arg": args.n_examples,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare] wrote meta -> {meta_path}")


if __name__ == "__main__":
    main()
