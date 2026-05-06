from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _safe_auc(y_true, y_score, kind: str) -> float:
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(set(y_true)) < 2:
        return float("nan")
    if kind == "roc":
        return float(roc_auc_score(y_true, y_score))
    if kind == "pr":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_dir", type=Path, required=True)
    ap.add_argument("--score_field", default="rel_prob_mean",
                    choices=["rel_prob_mean", "rel_prob_max"])
    ap.add_argument("--top_failures", type=int, default=20)
    args = ap.parse_args()

    scored_dir = args.scored_dir
    chunks_path = scored_dir / "chunks_scored.parquet"
    examples_path = scored_dir / "examples_scored.parquet"
    meta_path = scored_dir / "run_meta.json"
    if not chunks_path.exists() or not meta_path.exists():
        raise SystemExit(f"Missing inputs in {scored_dir}. Run score_dataset.py first.")

    import pandas as pd
    chunks = pd.read_parquet(chunks_path)
    examples = pd.read_parquet(examples_path)
    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    threshold = float(run_meta["thresholds"]["relevance"])
    threshold_source = run_meta["threshold_source"]
    print(f"[analyze] threshold={threshold} (source={threshold_source}); "
          f"score_field={args.score_field}; n_chunks_raw={len(chunks)}")

    chunks.to_csv(scored_dir / "chunks_scored.csv", index=False)

    n_chunks_raw = int(len(chunks))
    valid_mask = (
        chunks["rel_prob_mean"].notna()
        & chunks["rel_prob_max"].notna()
        & chunks["util_prob_mean"].notna()
        & chunks["util_prob_max"].notna()
    )
    n_dropped_nan = int((~valid_mask).sum())
    if n_dropped_nan:
        print(f"[analyze] dropping {n_dropped_nan}/{n_chunks_raw} chunk rows with NaN probs "
              "(empty tokenisation)")
    chunks = chunks[valid_mask].reset_index(drop=True)

    y_true = chunks["gt_is_relevant"].astype(bool).to_numpy()
    score = chunks[args.score_field].astype(float).to_numpy()
    y_pred = score > threshold

    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
        matthews_corrcoef,
        confusion_matrix,
        roc_curve,
        precision_recall_curve,
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[False, True]).tolist()

    roc_auc_mean = _safe_auc(y_true, chunks["rel_prob_mean"].astype(float).to_numpy(), "roc")
    roc_auc_max = _safe_auc(y_true, chunks["rel_prob_max"].astype(float).to_numpy(), "roc")
    pr_auc_mean = _safe_auc(y_true, chunks["rel_prob_mean"].astype(float).to_numpy(), "pr")
    pr_auc_max = _safe_auc(y_true, chunks["rel_prob_max"].astype(float).to_numpy(), "pr")

    examples_proc = examples[~examples["skipped"].astype(bool)].copy()
    top1_hits, topk_recalls, overlap_ratios = [], [], []
    for ex_id, group in chunks.groupby("example_id"):
        g = group.sort_values("chunk_id")
        gt_idx = set(g.loc[g["gt_is_relevant"], "chunk_id"].tolist())
        if not gt_idx:
            continue
        order = g.sort_values(args.score_field, ascending=False)
        top1 = int(order.iloc[0]["chunk_id"])
        top1_hits.append(top1 in gt_idx)
        k = len(gt_idx)
        topk = set(order.head(k)["chunk_id"].tolist())
        topk_recalls.append(len(topk & gt_idx) / k)
        pred_set = set(g.loc[g[args.score_field] > threshold, "chunk_id"].tolist())
        overlap_ratios.append(len(pred_set & gt_idx) / len(gt_idx))

    top1_acc = float(np.mean(top1_hits)) if top1_hits else float("nan")
    topk_recall = float(np.mean(topk_recalls)) if topk_recalls else float("nan")
    overlap_mean = float(np.mean(overlap_ratios)) if overlap_ratios else float("nan")

    summary = {
        "model_key": run_meta["model_key"],
        "max_length": run_meta["max_length"],
        "threshold": threshold,
        "threshold_source": threshold_source,
        "score_field": args.score_field,
        "n_chunks": int(len(chunks)),
        "n_chunks_raw": n_chunks_raw,
        "n_chunks_dropped_nan": n_dropped_nan,
        "n_examples_processed": int(run_meta["n_processed"]),
        "n_examples_skipped": int(run_meta["n_skipped"]),
        "n_examples_truncated": int(run_meta["n_truncated"]),
        "n_chunks_relevant": int(int(y_true.sum())),
        "n_chunks_irrelevant": int(int((~y_true).sum())),
        "accuracy": accuracy,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "mcc": mcc,
        "confusion_matrix": {"labels": [False, True], "matrix": cm},
        "roc_auc_mean": roc_auc_mean,
        "roc_auc_max": roc_auc_max,
        "pr_auc_mean": pr_auc_mean,
        "pr_auc_max": pr_auc_max,
        "top1_acc": top1_acc,
        "topk_recall": topk_recall,
        "overlap_ratio_mean": overlap_mean,
    }
    (scored_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[analyze] wrote metrics_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "confusion_matrix"}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 4))
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred=False", "pred=True"])
        ax.set_yticklabels(["gt=False", "gt=True"])
        for (i, j), val in np.ndenumerate(cm_arr):
            ax.text(j, i, int(val), ha="center", va="center",
                    color="white" if val > cm_arr.max()/2 else "black")
        ax.set_title(f"Confusion matrix (thr={threshold})")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(scored_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, chunks["rel_prob_mean"])
            ax.plot(fpr, tpr, label=f"mean (AUC={roc_auc_mean:.3f})")
            fpr2, tpr2, _ = roc_curve(y_true, chunks["rel_prob_max"])
            ax.plot(fpr2, tpr2, label=f"max  (AUC={roc_auc_max:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(scored_dir / "roc_curve.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        if len(set(y_true)) > 1:
            pr1, rc1, _ = precision_recall_curve(y_true, chunks["rel_prob_mean"])
            ax.plot(rc1, pr1, label=f"mean (AP={pr_auc_mean:.3f})")
            pr2, rc2, _ = precision_recall_curve(y_true, chunks["rel_prob_max"])
            ax.plot(rc2, pr2, label=f"max  (AP={pr_auc_max:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(scored_dir / "pr_curve.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 4))
        bins = np.linspace(0, 1, 31)
        ax.hist(score[y_true], bins=bins, alpha=0.5, label="gt=True", color="C1")
        ax.hist(score[~y_true], bins=bins, alpha=0.5, label="gt=False", color="C0")
        ax.axvline(threshold, color="red", linestyle="--", label=f"thr={threshold:.2f}")
        ax.set_xlabel(args.score_field); ax.set_ylabel("count")
        ax.set_title(f"{args.score_field} by GT")
        ax.legend()
        fig.tight_layout()
        fig.savefig(scored_dir / "prob_distribution_by_gt.png", dpi=120)
        plt.close(fig)
        print(f"[analyze] wrote plots: confusion_matrix.png, roc_curve.png, pr_curve.png, "
              f"prob_distribution_by_gt.png")
    except Exception as e:
        print(f"[analyze] plotting skipped: {e}")

    fp = chunks[(~chunks["gt_is_relevant"]) & y_pred].copy()
    fn = chunks[chunks["gt_is_relevant"] & (~y_pred)].copy()
    fp = fp.sort_values(args.score_field, ascending=False).head(args.top_failures)
    fn = fn.sort_values(args.score_field, ascending=True).head(args.top_failures)

    def _rows(df) -> List[Dict[str, Any]]:
        out = []
        for _, r in df.iterrows():
            out.append({
                "example_id": r["example_id"],
                "chunk_key": r["chunk_key"],
                "gt_is_relevant": bool(r["gt_is_relevant"]),
                "rel_prob_mean": float(r["rel_prob_mean"]),
                "rel_prob_max": float(r["rel_prob_max"]),
                "chunk_text": r["chunk_text"],
            })
        return out

    failure = {
        "threshold": threshold,
        "score_field": args.score_field,
        "false_positives_top": _rows(fp),
        "false_negatives_top": _rows(fn),
    }
    (scored_dir / "failure_cases.json").write_text(
        json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[analyze] wrote failure_cases.json (FP={len(fp)}, FN={len(fn)})")


if __name__ == "__main__":
    main()
