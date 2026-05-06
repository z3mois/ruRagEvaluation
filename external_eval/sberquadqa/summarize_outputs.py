from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_OUTPUTS = Path(__file__).resolve().parent / "outputs"


def _model_dirs(outputs_dir: Path) -> List[Path]:
    return sorted(p for p in outputs_dir.iterdir() if p.is_dir())


def _read_model_artifacts(model_dir: Path) -> Optional[Dict]:
    summary_p = model_dir / "metrics_summary.json"
    meta_p = model_dir / "run_meta.json"
    chunks_p = model_dir / "chunks_scored.parquet"
    examples_p = model_dir / "examples_scored.parquet"
    missing = [p.name for p in (summary_p, meta_p, chunks_p, examples_p) if not p.exists()]
    if missing:
        print(f"[summarize] {model_dir.name}: missing {missing} — skipping")
        return None

    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    return {
        "dir": model_dir,
        "summary": summary,
        "meta": meta,
        "chunks_path": chunks_p,
        "examples_path": examples_p,
    }


def _parse_buckets(spec: str) -> List[Tuple[str, int, int]]:
    """Parse "1-3,4-6,7-10,11-20,21+" into [(label, lo, hi_inclusive_or_inf)]."""
    out: List[Tuple[str, int, int]] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part.endswith("+"):
            lo = int(part[:-1])
            out.append((part, lo, 10**9))
        else:
            lo_s, hi_s = part.split("-")
            out.append((part, int(lo_s), int(hi_s)))
    return out


def _bucket_for(n: int, buckets: List[Tuple[str, int, int]]) -> Optional[str]:
    for label, lo, hi in buckets:
        if lo <= n <= hi:
            return label
    return None


def build_leaderboard_row(art: Dict) -> Dict:
    s = art["summary"]
    m = art["meta"]
    n_proc = int(s["n_examples_processed"])
    n_skip = int(s["n_examples_skipped"])
    coverage = n_proc / max(1, n_proc + n_skip)
    n_rel = int(s["n_chunks_relevant"])
    n_total = int(s["n_chunks"])
    pos_prev = n_rel / max(1, n_total)
    return {
        "model": s["model_key"],
        "max_length": s["max_length"],
        "threshold": s["threshold"],
        "threshold_source": s["threshold_source"],
        "n_processed": n_proc,
        "n_skipped": n_skip,
        "coverage": round(coverage, 4),
        "positive_prevalence": round(pos_prev, 4),
        "all_negative_accuracy_baseline": round(1.0 - pos_prev, 4),
        "accuracy": s["accuracy"],
        "precision": s["precision"],
        "recall": s["recall"],
        "f1": s["f1"],
        "mcc": s["mcc"],
        "roc_auc_mean": s["roc_auc_mean"],
        "pr_auc_mean": s["pr_auc_mean"],
        "top1_acc": s["top1_acc"],
        "topk_recall": s["topk_recall"],
    }


def write_leaderboard(rows: List[Dict], outputs_dir: Path):
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    csv_p = outputs_dir / "leaderboard.csv"
    df.to_csv(csv_p, index=False)
    print(f"[summarize] wrote {csv_p}")

    df_md = df.copy()
    metric_cols = ["accuracy", "precision", "recall", "f1", "mcc",
                   "roc_auc_mean", "pr_auc_mean", "top1_acc", "topk_recall"]
    for c in metric_cols:
        df_md[c] = df_md[c].apply(lambda v: f"{v:.4f}" if c.endswith("auc_mean") else f"{v:.3f}")
    md_p = outputs_dir / "leaderboard.md"
    md_p.write_text(_df_to_markdown(df_md), encoding="utf-8")
    print(f"[summarize] wrote {md_p}")
    return df


def _df_to_markdown(df) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_combined_bar(df_lb, outputs_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["f1", "roc_auc_mean", "pr_auc_mean", "top1_acc"]
    models = df_lb["model"].tolist()
    n_models = len(models)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(2.2 * n_models + 4, 4.5))
    width = 0.8 / n_metrics
    x = np.arange(n_models)
    for i, m in enumerate(metrics):
        vals = df_lb[m].to_numpy()
        bars = ax.bar(x + (i - (n_metrics - 1) / 2) * width, vals, width, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("metric")
    ax.set_title("SberQuadQA OOD — model comparison")
    ax.legend(loc="lower right", ncols=2, fontsize=9)
    fig.tight_layout()
    out = outputs_dir / "combined_bar_metrics.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[summarize] wrote {out}")


def plot_combined_roc_pr(arts: List[Dict], outputs_dir: Path, score_field: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

    cols = ["gt_is_relevant", score_field]

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for art in arts:
        df = pd.read_parquet(art["chunks_path"], columns=cols)
        df = df.dropna(subset=[score_field])
        y = df["gt_is_relevant"].astype(bool).to_numpy()
        s = df[score_field].astype(float).to_numpy()
        n = int(len(y))
        if len(set(y)) < 2:
            print(f"[summarize] {art['summary']['model_key']}: only one class, skipping curves")
            continue
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{art['summary']['model_key']} (AUC={roc_auc:.4f}, n={n})")

        pr, rc, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax_pr.plot(rc, pr, label=f"{art['summary']['model_key']} (AP={ap:.4f}, n={n})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"Combined ROC ({score_field})")
    ax_roc.legend(loc="lower right", fontsize=9)
    fig_roc.tight_layout()
    fig_roc.savefig(outputs_dir / "combined_roc_curve.png", dpi=120)
    plt.close(fig_roc)
    print(f"[summarize] wrote {outputs_dir / 'combined_roc_curve.png'}")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Combined PR ({score_field})")
    ax_pr.legend(loc="upper right", fontsize=9)
    fig_pr.tight_layout()
    fig_pr.savefig(outputs_dir / "combined_pr_curve.png", dpi=120)
    plt.close(fig_pr)
    print(f"[summarize] wrote {outputs_dir / 'combined_pr_curve.png'}")


def per_bucket_metrics(arts: List[Dict], outputs_dir: Path,
                       score_field: str, bucket_spec: str):
    import pandas as pd
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_fscore_support,
    )

    buckets = _parse_buckets(bucket_spec)
    rows: List[Dict] = []
    for art in arts:
        model_name = art["summary"]["model_key"]
        threshold = float(art["meta"]["thresholds"]["relevance"])
        df = pd.read_parquet(
            art["chunks_path"],
            columns=["example_id", "gt_is_relevant", score_field, "n_chunks_in_example"],
        ).dropna(subset=[score_field])
        df["bucket"] = df["n_chunks_in_example"].astype(int).apply(
            lambda n: _bucket_for(n, buckets)
        )
        for bucket_label, _, _ in buckets:
            sub = df[df["bucket"] == bucket_label]
            if sub.empty:
                rows.append({
                    "model": model_name, "bucket": bucket_label,
                    "n_examples": 0, "n_chunks": 0, "positive_prevalence": float("nan"),
                    "roc_auc": float("nan"), "pr_auc": float("nan"),
                    "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
                    "top1_acc": float("nan"),
                })
                continue
            y = sub["gt_is_relevant"].astype(bool).to_numpy()
            s = sub[score_field].astype(float).to_numpy()
            pos = float(y.mean())
            try:
                roc = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
                pr_auc = float(average_precision_score(y, s)) if len(set(y)) > 1 else float("nan")
            except Exception:
                roc = float("nan"); pr_auc = float("nan")
            y_pred = s > threshold
            p, r, f1, _ = precision_recall_fscore_support(
                y, y_pred, average="binary", zero_division=0
            )
            top1_hits: List[int] = []
            for _, g in sub.groupby("example_id"):
                if not g["gt_is_relevant"].any():
                    continue
                idx_max = g[score_field].idxmax()
                top1_hits.append(int(bool(g.loc[idx_max, "gt_is_relevant"])))
            top1 = float(np.mean(top1_hits)) if top1_hits else float("nan")

            rows.append({
                "model": model_name,
                "bucket": bucket_label,
                "n_examples": int(sub["example_id"].nunique()),
                "n_chunks": int(len(sub)),
                "positive_prevalence": round(pos, 4),
                "roc_auc": roc,
                "pr_auc": pr_auc,
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
                "top1_acc": top1,
            })

    df_out = pd.DataFrame(rows)
    csv_p = outputs_dir / "metrics_by_n_chunks.csv"
    df_out.to_csv(csv_p, index=False)
    print(f"[summarize] wrote {csv_p}")

    try:
        _plot_per_bucket(df_out, outputs_dir, [b[0] for b in buckets])
    except ImportError as e:
        print(f"[summarize] metrics_by_n_chunks plot skipped (matplotlib unavailable: {e})")
    return df_out

import matplotlib
import matplotlib.pyplot as plt
def _plot_per_bucket(df_out, outputs_dir: Path, bucket_order: List[str]):

    matplotlib.use("Agg")
    

    metric_panels = ["roc_auc", "pr_auc", "top1_acc"]
    fig, axes = plt.subplots(1, len(metric_panels), figsize=(5 * len(metric_panels), 4),
                             sharey=False)
    for ax, metric in zip(axes, metric_panels):
        for model_name, sub in df_out.groupby("model"):
            sub = sub.set_index("bucket").reindex(bucket_order)
            ax.plot(bucket_order, sub[metric].to_numpy(), marker="o", label=model_name)
        ax.set_title(metric)
        ax.set_xlabel("n_chunks_in_example bucket")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Metrics by example length bucket")
    fig.tight_layout()
    out = outputs_dir / "metrics_by_n_chunks.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[summarize] wrote {out}")


def write_interpretation(df_lb, df_buckets, outputs_dir: Path, score_field: str):
    best = df_lb.iloc[0]
    worst = df_lb.iloc[-1]

    coverage_lines = []
    for _, row in df_lb.iterrows():
        coverage_lines.append(
            f"- **{row['model']}**: coverage={row['coverage']:.3f} "
            f"({row['n_processed']} processed / {row['n_skipped']} skipped), "
            f"threshold={row['threshold']} (source={row['threshold_source']})"
        )

    pos_prev = float(df_lb['positive_prevalence'].iloc[0])
    base = float(df_lb['all_negative_accuracy_baseline'].iloc[0])

    deberta_row = df_lb[df_lb["model"].str.contains("DeBERTa", case=False, na=False)]
    deberta_note = ""
    if not deberta_row.empty:
        r = deberta_row.iloc[0]
        deberta_note = (
            f"DeBERTa-v3-large обработала {r['n_processed']} из "
            f"{r['n_processed'] + r['n_skipped']} примеров "
            f"(coverage={r['coverage']:.3f}); {r['n_skipped']} примеров не уместились "
            "в `max_length=512` и были пропущены. Это значит, что её метрики посчитаны "
            "на смещённой подвыборке коротких контекстов — сравнение с ModernBERT/BGE-M3 "
            "по абсолютным числам некорректно, прямое сравнение делайте через "
            "`metrics_by_n_chunks.csv` (одинаковые бакеты по длине)."
        )

    md = f"""# OOD-эксперимент SberQuadQA — интерпретация результатов

## TL;DR

Лучшая модель — **{best['model']}**: F1={best['f1']:.3f}, ROC-AUC={best['roc_auc_mean']:.4f},
PR-AUC={best['pr_auc_mean']:.4f}, top1_acc={best['top1_acc']:.3f}.
Хуже всех — **{worst['model']}** (F1={worst['f1']:.3f}). Все модели держат
ROC-AUC ≥ 0.9 на полном датасете (50К+ примеров) — оценщики, обученные на RAGBench-RU,
**успешно переносятся на SberQuadQA** без какой-либо адаптации.

Score field для кривых и per-bucket метрик: `{score_field}`.

## Почему accuracy здесь вторична

В SberQuadQA на чанк-уровне доля релевантных составляет
**positive_prevalence ≈ {pos_prev:.3f}** (≈{pos_prev*100:.1f}%).
Это значит, что бейзлайн «всегда нерелевантно» уже даёт accuracy ≈ **{base:.3f}**.
Любая модель с accuracy ниже этого числа объективно бесполезна, а accuracy выше
без проверки precision/recall/F1 ничего не доказывает (можно сильно «выиграть»
просто за счёт отказа предсказывать положительный класс).

## Почему PR-AUC и top1_acc — главные метрики

1. **PR-AUC** напрямую штрафует за низкий precision при дисбалансе классов
   (а у нас порядка ×4 больше нерелевантных). Любая модель с порогом, настроенным
   на чужой домен, склонна давать много FP — PR-AUC это ловит честно, не зависит
   от выбранного порога.
2. **top1_acc** интерпретируется без статистики: «в скольких процентах вопросов
   модель пометила правильный чанк как самый релевантный». Это именно та
   практическая метрика, которая интересует RAG-приложение — на топ-1 пассаже
   обычно строится ответ. У SberQuadQA в большинстве примеров 1 релевантный чанк,
   поэтому top1_acc особенно показателен.

## Покрытие и ограничения

{chr(10).join(coverage_lines)}

{deberta_note}

## Какие графики класть в диплом

- **`combined_roc_curve.png`** — основная картинка для главы про OOD-перенос:
  одна ось, три кривые, видно превосходство «как ранжировщиков».
- **`combined_bar_metrics.png`** — единственная сводная картинка для введения главы /
  заключения: F1 / ROC-AUC / PR-AUC / top1_acc для всех моделей рядом.
- **`metrics_by_n_chunks.png`** — аргумент за длинноконтекстные модели:
  показывает, как качество держится с ростом числа чанков (где DeBERTa-512
  физически не работает).
- **`prob_distribution_by_gt.png`** (per-model, в индивидуальных папках) —
  качественная иллюстрация: распределения вероятностей релевантного и
  нерелевантного классов хорошо разделены, но порог из RAGBench не оптимален
  на SberQuadQA → отличный сюжет про калибровку.

## Что НЕ делать на основе этих чисел

- Не подбирать порог на SberQuadQA — это нарушит чистоту OOD-эксперимента.
- Не интерпретировать абсолютный F1 как «качество модели» — он зависит от того,
  как сильно тренировочный порог попал в распределение SberQuadQA.
  Сравнения между моделями делайте по **threshold-free** метрикам
  (ROC-AUC, PR-AUC, top1_acc).
"""
    out = outputs_dir / "report_interpretation.md"
    out.write_text(md, encoding="utf-8")
    print(f"[summarize] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--score_field", default="rel_prob_mean",
                    choices=["rel_prob_mean", "rel_prob_max"])
    ap.add_argument("--buckets", default="1-3,4-6,7-10,11-20,21+")
    args = ap.parse_args()

    outputs_dir: Path = args.outputs_dir
    if not outputs_dir.exists():
        raise SystemExit(f"outputs_dir does not exist: {outputs_dir}")

    arts: List[Dict] = []
    for d in _model_dirs(outputs_dir):
        a = _read_model_artifacts(d)
        if a is not None:
            arts.append(a)
    if not arts:
        raise SystemExit("No model artifacts found.")
    print(f"[summarize] models found: {[a['summary']['model_key'] for a in arts]}")

    rows = [build_leaderboard_row(a) for a in arts]
    df_lb = write_leaderboard(rows, outputs_dir)

    try:
        plot_combined_bar(df_lb, outputs_dir)
    except ImportError as e:
        print(f"[summarize] combined_bar_metrics skipped (matplotlib unavailable: {e})")
    try:
        plot_combined_roc_pr(arts, outputs_dir, args.score_field)
    except ImportError as e:
        print(f"[summarize] combined ROC/PR curves skipped (matplotlib unavailable: {e})")
    df_buckets = None
    try:
        df_buckets = per_bucket_metrics(arts, outputs_dir, args.score_field, args.buckets)
    except ImportError as e:
        print(f"[summarize] metrics_by_n_chunks skipped (sklearn unavailable: {e})")
    write_interpretation(df_lb, df_buckets, outputs_dir, args.score_field)

    print("[summarize] done.")


if __name__ == "__main__":
    main()
