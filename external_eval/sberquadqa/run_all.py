"""Run score + analyze for every model in MODEL_REGISTRY whose checkpoint exists.

Usage:
    python -m external_eval.sberquadqa.run_all
    python -m external_eval.sberquadqa.run_all --limit 10  # smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from ._common import MODEL_REGISTRY, preflight_check, safe_model_tag

OUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_DATA = Path(__file__).resolve().parent / "cache" / "sberquadqa_prepared.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--score_field", default="rel_prob_mean")
    ap.add_argument("--truncate", action="store_true")
    args = ap.parse_args()

    print("[run_all] preflight for all models in MODEL_REGISTRY:")
    runnable: List[str] = []
    for key in MODEL_REGISTRY:
        pre = preflight_check(key)
        marker = "OK " if pre.ok else "SKIP"
        print(f"  [{marker}] {key}: {pre.message}")
        if pre.ok:
            runnable.append(key)
    if not runnable:
        raise SystemExit("[run_all] no checkpoints available — nothing to run.")

    leaderboard: List[Dict] = []
    for key in runnable:
        max_length = MODEL_REGISTRY[key]["max_length"]
        out_dir = OUT_ROOT / safe_model_tag(key, max_length)
        out_dir.mkdir(parents=True, exist_ok=True)
        score_cmd = [
            sys.executable, "-m", "external_eval.sberquadqa.score_dataset",
            "--model_key", key,
            "--data", str(args.data),
            "--device", args.device,
            "--out_dir", str(out_dir),
        ]
        if args.limit is not None:
            score_cmd += ["--limit", str(args.limit)]
        if args.truncate:
            score_cmd += ["--truncate"]
        print(f"[run_all] >>> {' '.join(score_cmd)}")
        rc = subprocess.call(score_cmd)
        if rc != 0:
            print(f"[run_all] score_dataset failed for '{key}' (rc={rc}), skipping analyze.")
            continue

        analyze_cmd = [
            sys.executable, "-m", "external_eval.sberquadqa.analyze_results",
            "--scored_dir", str(out_dir),
            "--score_field", args.score_field,
        ]
        print(f"[run_all] >>> {' '.join(analyze_cmd)}")
        rc = subprocess.call(analyze_cmd)
        if rc != 0:
            print(f"[run_all] analyze_results failed for '{key}' (rc={rc}).")
            continue

        summary = json.loads((out_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        leaderboard.append({
            "model": summary["model_key"],
            "max_length": summary["max_length"],
            "threshold": summary["threshold"],
            "threshold_source": summary["threshold_source"],
            "accuracy": summary["accuracy"],
            "precision": summary["precision"],
            "recall": summary["recall"],
            "f1": summary["f1"],
            "mcc": summary["mcc"],
            "roc_auc_mean": summary["roc_auc_mean"],
            "roc_auc_max": summary["roc_auc_max"],
            "pr_auc_mean": summary["pr_auc_mean"],
            "pr_auc_max": summary["pr_auc_max"],
            "top1_acc": summary["top1_acc"],
            "n_processed": summary["n_examples_processed"],
            "n_skipped": summary["n_examples_skipped"],
        })

    if leaderboard:
        import pandas as pd
        df = pd.DataFrame(leaderboard)
        out = OUT_ROOT / "leaderboard.csv"
        df.to_csv(out, index=False)
        print(f"[run_all] wrote {out}")
        print(df.to_string(index=False))
    else:
        print("[run_all] no successful runs — leaderboard not written.")


if __name__ == "__main__":
    main()
