from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import pandas as pd


@dataclass
class ExperimentResult:
    model_name: str
    max_length: Optional[int]
    output_dir: str
    val_metrics: dict
    best_val_f1: float
    train_sizes: dict
    cfg_dict: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=lambda: {"relevance": 0.5, "utilization": 0.5, "adherence": 0.5})
    run_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def save_result(result: ExperimentResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    print(f"[results] Saved result to: {path}")


def load_result(path: str) -> ExperimentResult:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return ExperimentResult(**d)


def load_results_from_dirs(sweep_root: str) -> pd.DataFrame:
    """Scan sweep_root/**/result.json and return a DataFrame of all experiment results.

    Replaces the manual pkl-file index-alignment pattern from the notebook.
    Usage::

        df = load_results_from_dirs("sweeps")
        df.sort_values("relevance_f1", ascending=False)
    """
    rows = []
    for result_path in sorted(Path(sweep_root).rglob("result.json")):
        try:
            result = load_result(str(result_path))
            row = {
                "run_id": result.run_id,
                "model_name": result.model_name,
                "max_length": result.max_length,
                "output_dir": result.output_dir,
                "best_val_f1": result.best_val_f1,
                "timestamp": result.timestamp,
            }
            row.update(result.val_metrics)
            row.update({f"test_{k}": v for k, v in result.test_metrics.items()})
            row["threshold_relevance"]   = result.thresholds.get("relevance",   0.5)
            row["threshold_utilization"] = result.thresholds.get("utilization", 0.5)
            row["threshold_adherence"]   = result.thresholds.get("adherence",   0.5)
            row.update({f"n_{k}": v for k, v in result.train_sizes.items()})
            rows.append(row)
        except Exception as e:
            print(f"[results] Warning: could not load {result_path}: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
