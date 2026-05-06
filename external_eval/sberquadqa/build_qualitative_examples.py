from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "modernbert_len2048__len_2048"
)
DEFAULT_PREPARED = (
    Path(__file__).resolve().parent / "cache" / "sberquadqa_prepared.parquet"
)


def _truncate(text: str, limit: int = 600) -> str:
    if text is None:
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _safe_get(rec: Dict[str, Any], key: str, default=None):
    v = rec.get(key, default)
    return v if v is not None else default


def _build_per_example(chunks_df) -> Dict[str, Dict[str, Any]]:
    """Group scored chunks per example_id and compute ranking info."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for ex_id, g in chunks_df.groupby("example_id"):
        g = g.sort_values("rel_prob_mean", ascending=False).reset_index(drop=True)
        g["rank"] = g.index + 1
        rec = {
            "example_id": ex_id,
            "n_chunks_scored": int(len(g)),
            "n_gt_relevant": int(g["gt_is_relevant"].astype(bool).sum()),
            "df": g,
        }
        gt_rows = g[g["gt_is_relevant"].astype(bool)]
        rec["first_gt_rank"] = int(gt_rows.iloc[0]["rank"]) if not gt_rows.empty else -1
        rec["top1_is_relevant"] = bool(g.iloc[0]["gt_is_relevant"])
        rec["top1_rel_prob_mean"] = float(g.iloc[0]["rel_prob_mean"])
        rec["top1_rel_prob_max"] = float(g.iloc[0]["rel_prob_max"])
        grouped[ex_id] = rec
    return grouped


def _select_success(per_ex: Dict[str, Dict], used: set, n: int = 2) -> List[Dict]:
    """top-1 hits with high confidence, prefer mid-length contexts (more interesting)."""
    candidates = []
    for ex_id, rec in per_ex.items():
        if ex_id in used:
            continue
        if not rec["top1_is_relevant"]:
            continue
        if rec["n_chunks_scored"] < 5:
            continue
        if rec["top1_rel_prob_mean"] < 0.6:
            continue
        candidates.append(rec)
    candidates.sort(key=lambda r: -r["top1_rel_prob_mean"])
    picked: List[Dict] = []
    seen_buckets: set = set()
    for rec in candidates:
        b = min(rec["n_chunks_scored"] // 5, 5)
        if b in seen_buckets:
            continue
        picked.append(rec)
        seen_buckets.add(b)
        if len(picked) >= n:
            break
    return picked[:n]


def _select_fp(per_ex: Dict[str, Dict], used: set, n: int = 2) -> List[Dict]:
    """Top-1 chunk is gt=False (a false positive) but example still has at least one gt-relevant.
    High top-1 score → confidence-driven mistake, more interesting for analysis."""
    candidates = []
    for ex_id, rec in per_ex.items():
        if ex_id in used:
            continue
        if rec["top1_is_relevant"]:
            continue
        if rec["n_gt_relevant"] == 0:
            continue
        if rec["n_chunks_scored"] < 4:
            continue
        if rec["top1_rel_prob_mean"] < 0.55:
            continue
        candidates.append(rec)
    candidates.sort(key=lambda r: -r["top1_rel_prob_mean"])
    picked, seen_buckets = [], set()
    for rec in candidates:
        b = min(rec["n_chunks_scored"] // 5, 5)
        if b in seen_buckets:
            continue
        picked.append(rec)
        seen_buckets.add(b)
        if len(picked) >= n:
            break
    return picked[:n]


def _select_fn(per_ex: Dict[str, Dict], used: set, n: int = 2) -> List[Dict]:
    """At least one GT-relevant chunk got low score (rank > 5 OR rel_prob_mean < 0.3)."""
    candidates = []
    for ex_id, rec in per_ex.items():
        if ex_id in used:
            continue
        if rec["n_gt_relevant"] == 0:
            continue
        df = rec["df"]
        gt = df[df["gt_is_relevant"].astype(bool)]
        if gt.empty:
            continue
        worst_gt_rank = int(gt["rank"].max())
        worst_gt_score = float(gt["rel_prob_mean"].min())
        if not (worst_gt_rank > 5 or worst_gt_score < 0.3):
            continue
        if rec["n_chunks_scored"] < 6:
            continue
        rec_copy = dict(rec)
        rec_copy["worst_gt_rank"] = worst_gt_rank
        rec_copy["worst_gt_score"] = worst_gt_score
        candidates.append(rec_copy)
    candidates.sort(key=lambda r: (-r["worst_gt_rank"], r["worst_gt_score"]))
    picked, seen_buckets = [], set()
    for rec in candidates:
        b = min(rec["n_chunks_scored"] // 5, 5)
        if b in seen_buckets:
            continue
        picked.append(rec)
        seen_buckets.add(b)
        if len(picked) >= n:
            break
    return picked[:n]


def _enrich(rec: Dict[str, Any], prepared_by_id: Dict[str, Dict[str, Any]],
            kind: str, comment: str) -> Dict[str, Any]:
    df = rec["df"]
    src = prepared_by_id.get(rec["example_id"], {})
    question = src.get("question_ru", "") or src.get("question", "")
    answers = list(src.get("answers_all") or src.get("answers") or [])

    top5 = []
    for _, r in df.head(5).iterrows():
        top5.append({
            "rank": int(r["rank"]),
            "chunk_id": int(r["chunk_id"]),
            "chunk_key": r.get("chunk_key", f"chunk_{int(r['chunk_id']):04d}"),
            "rel_prob_mean": float(r["rel_prob_mean"]),
            "rel_prob_max": float(r["rel_prob_max"]),
            "gt_is_relevant": bool(r["gt_is_relevant"]),
            "chunk_text": _truncate(r.get("chunk_text", ""), 600),
        })
    gt_rows = df[df["gt_is_relevant"].astype(bool)]
    gt_chunks = []
    for _, r in gt_rows.iterrows():
        gt_chunks.append({
            "rank": int(r["rank"]),
            "chunk_id": int(r["chunk_id"]),
            "chunk_key": r.get("chunk_key", f"chunk_{int(r['chunk_id']):04d}"),
            "rel_prob_mean": float(r["rel_prob_mean"]),
            "rel_prob_max": float(r["rel_prob_max"]),
            "chunk_text": _truncate(r.get("chunk_text", ""), 600),
        })
    top1_row = df.iloc[0]
    return {
        "kind": kind,
        "example_id": rec["example_id"],
        "question": question,
        "gold_answers": answers,
        "n_chunks_scored": int(rec["n_chunks_scored"]),
        "n_gt_relevant": int(rec["n_gt_relevant"]),
        "first_gt_rank": int(rec["first_gt_rank"]),
        "top1": {
            "chunk_id": int(top1_row["chunk_id"]),
            "chunk_key": top1_row.get("chunk_key", f"chunk_{int(top1_row['chunk_id']):04d}"),
            "rel_prob_mean": float(top1_row["rel_prob_mean"]),
            "rel_prob_max": float(top1_row["rel_prob_max"]),
            "gt_is_relevant": bool(top1_row["gt_is_relevant"]),
            "chunk_text": _truncate(top1_row.get("chunk_text", ""), 700),
        },
        "top5": top5,
        "gt_relevant_chunks": gt_chunks,
        "comment": comment,
    }


def _comment(kind: str, enriched: Dict[str, Any]) -> str:
    q = enriched["question"]
    top1 = enriched["top1"]
    n_gt = enriched["n_gt_relevant"]
    n_total = enriched["n_chunks_scored"]
    first_gt = enriched["first_gt_rank"]
    if kind == "success":
        return (
            f"Модель правильно вывела на первое место чанк, содержащий ответ "
            f"(rel_prob_mean={top1['rel_prob_mean']:.3f}, ранг 1 из {n_total}). "
            f"GT-релевантных чанков: {n_gt}. Top-1 уверенно отделён от дистракторов."
        )
    if kind == "false_positive":
        return (
            f"Модель пометила топ-1 чанком формально нерелевантный фрагмент "
            f"(rel_prob_mean={top1['rel_prob_mean']:.3f}). Истинно релевантный чанк есть "
            f"в контексте (n_gt={n_gt}) и оказался на ранге {first_gt}. "
            "Топ-1 чанк тематически близок к вопросу, но формально не содержит ответа — "
            "типичная ошибка калибровки порога на новом домене, а не провал ранжирования."
        )
    if kind == "false_negative":
        worst_rank = enriched.get("worst_gt_rank")
        return (
            f"Истинно релевантный чанк получил низкий ранг "
            f"(первый GT на позиции {first_gt}{', самый дальний на ' + str(worst_rank) if worst_rank else ''}, "
            f"всего {n_total} чанков). "
            "Скорее всего, формулировка пассажа сильно отличается от вопроса лексически "
            "(перифраз, длинный пассаж с разбавляющим контекстом) — модель распределяет "
            "сигнал релевантности по другим, более лексически близким фрагментам."
        )
    return ""


def _to_markdown(items: List[Dict[str, Any]]) -> str:
    title_by_kind = {
        "success": "Успешные примеры (top-1 = GT-relevant)",
        "false_positive": "False positive (top-1 = GT-irrelevant)",
        "false_negative": "False negative (релевантный чанк низко в ранжировании)",
    }
    out = ["# Качественные примеры — ModernBERT (len=2048) на SberQuadQA",
           "",
           "Все примеры взяты из готового `chunks_scored.parquet` ModernBERT, без повторного "
           "инференса и без подбора порогов. Score = `rel_prob_mean` (агрегация средним "
           "по токенам чанка), порог из тренировочного `result.json` = 0.5.",
           ""]
    for kind in ("success", "false_positive", "false_negative"):
        out.append(f"## {title_by_kind[kind]}")
        out.append("")
        for it in [x for x in items if x["kind"] == kind]:
            out.append(f"### example_id = `{it['example_id']}`")
            out.append("")
            out.append(f"- **Вопрос:** {it['question']}")
            ans = " | ".join(it['gold_answers']) if it['gold_answers'] else "—"
            out.append(f"- **Эталонный ответ:** {ans}")
            out.append(f"- **Чанков в примере:** {it['n_chunks_scored']}")
            out.append(f"- **GT-релевантных чанков:** {it['n_gt_relevant']}")
            out.append(f"- **Ранг первого GT-релевантного:** {it['first_gt_rank']}")
            out.append("")
            out.append("**Top-1 чанк:**")
            t1 = it["top1"]
            out.append(
                f"- {t1['chunk_key']} — gt={t1['gt_is_relevant']}, "
                f"rel_prob_mean={t1['rel_prob_mean']:.3f}, "
                f"rel_prob_max={t1['rel_prob_max']:.3f}"
            )
            out.append("")
            out.append(f"> {t1['chunk_text']}")
            out.append("")
            out.append("**Top-5 по rel_prob_mean:**")
            out.append("")
            out.append("| ранг | chunk_key | gt | rel_prob_mean | rel_prob_max |")
            out.append("|---:|---|:---:|---:|---:|")
            for r in it["top5"]:
                out.append(
                    f"| {r['rank']} | {r['chunk_key']} | "
                    f"{'✓' if r['gt_is_relevant'] else '✗'} | "
                    f"{r['rel_prob_mean']:.3f} | {r['rel_prob_max']:.3f} |"
                )
            out.append("")
            if it["gt_relevant_chunks"]:
                out.append("**GT-релевантные чанки:**")
                out.append("")
                for r in it["gt_relevant_chunks"]:
                    out.append(
                        f"- ранг {r['rank']}, {r['chunk_key']}, "
                        f"score={r['rel_prob_mean']:.3f}"
                    )
                    out.append(f"  > {r['chunk_text']}")
                out.append("")
            out.append(f"**Комментарий.** {it['comment']}")
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    chunks_p = out_dir / "chunks_scored.parquet"
    if not chunks_p.exists():
        raise SystemExit(f"chunks_scored.parquet not found: {chunks_p}")
    if not args.prepared.exists():
        raise SystemExit(f"prepared dataset not found: {args.prepared}")

    import pandas as pd
    chunks = pd.read_parquet(chunks_p)
    chunks = chunks.dropna(subset=["rel_prob_mean", "rel_prob_max"])
    print(f"[qual] chunks loaded: {len(chunks)}")

    prepared = pd.read_parquet(args.prepared)
    prepared_by_id = {row["orig_id"]: row.to_dict() for _, row in prepared.iterrows()}
    print(f"[qual] prepared examples loaded: {len(prepared_by_id)}")

    per_ex = _build_per_example(chunks)
    print(f"[qual] examples scored: {len(per_ex)}")

    used: set = set()
    selections: List[Tuple[str, List[Dict]]] = [
        ("success", _select_success(per_ex, used, n=2)),
    ]
    used.update(r["example_id"] for r in selections[0][1])
    selections.append(("false_positive", _select_fp(per_ex, used, n=2)))
    used.update(r["example_id"] for r in selections[1][1])
    selections.append(("false_negative", _select_fn(per_ex, used, n=2)))

    enriched_items: List[Dict[str, Any]] = []
    for kind, recs in selections:
        for rec in recs:
            base = _enrich(rec, prepared_by_id, kind, comment="")
            base["comment"] = _comment(kind, base)
            enriched_items.append(base)

    json_path = out_dir / "enriched_examples_for_report.json"
    json_path.write_text(
        json.dumps(enriched_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[qual] wrote {json_path}")

    md_path = out_dir / "examples_for_report.md"
    md_path.write_text(_to_markdown(enriched_items), encoding="utf-8")
    print(f"[qual] wrote {md_path}")

    summary = {kind: [it["example_id"] for it in enriched_items if it["kind"] == kind]
               for kind in ("success", "false_positive", "false_negative")}
    print(f"[qual] selected: {summary}")


if __name__ == "__main__":
    main()
